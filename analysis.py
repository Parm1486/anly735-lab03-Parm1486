"""Custom CivilComments population-shift lab; not the official WILDS protocol."""


from pathlib import Path
from itertools import combinations
from importlib.metadata import version
import argparse
import hashlib
import json
import platform
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline

IDENTITIES = ['male', 'female', 'LGBTQ', 'christian', 'muslim',
              'other_religions', 'black', 'white']
ENVIRONMENTS = ['validation', 'ID_test', 'shifted_test']
SCORES = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'specificity',
          'f1', 'roc_auc', 'average_precision', 'brier_score', 'log_loss', 'ece_10_bins']
BOOT_SCORES = ['accuracy', 'balanced_accuracy', 'precision', 'recall',
               'specificity', 'f1', 'brier_score', 'log_loss']
LOWER_BETTER = {'brier_score', 'log_loss', 'ece_10_bins'}
THRESHOLD = 0.5


def save_table(frame, output, name):
    frame.to_csv(output / f'{name}.csv', index=False)
    return frame


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(series):
    return series.str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()


def prepare_data(path, output):
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    required = ['id', 'article_id', 'comment_text', 'split', 'toxicity'] + IDENTITIES
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    df = pd.read_csv(path, usecols=required, low_memory=False)
    df['source_row'] = np.arange(len(df))
    raw_n = len(df)
    split_counts = df['split'].value_counts(dropna=False).rename_axis('split').reset_index(name='n')
    save_table(split_counts, output, 'source_split_counts')
    missing_counts = df.isna().sum().rename_axis('column').reset_index(name='missing_n')
    save_table(missing_counts, output, 'source_missing_values')
    invalid = df.comment_text.isna() | df.comment_text.fillna('').str.strip().eq('')
    save_table(df.loc[invalid, ['source_row', 'id', 'split']], output, 'excluded_text_records')
    excluded_n = int(invalid.sum())
    df = df.loc[~invalid].copy()
    for column in ['toxicity'] + IDENTITIES:
        df[column] = pd.to_numeric(df[column], errors='raise')
        if not df[column].between(0, 1).all():
            raise ValueError(f'Missing or invalid scores in {column}')
    if not df['split'].isin(['train', 'val', 'test']).all():
        raise ValueError('Unexpected or missing split names.')
    if df[['id', 'article_id']].isna().any().any():
        raise ValueError('Missing comment or article IDs.')
    df['y'] = (df.toxicity >= THRESHOLD).astype(int)
    present = df[IDENTITIES].ge(THRESHOLD).any(axis=1)
    df['environment'] = np.where(present, 'identity_present', 'identity_absent')
    population = df.groupby(['split', 'environment'], observed=True).agg(
        n=('y', 'size'), toxic_count=('y', 'sum'), toxicity_prevalence=('y', 'mean')).reset_index()
    save_table(population, output, 'population_summary')
    masks = {'train': df['split'].eq('train') & ~present,
             'validation': df['split'].eq('val') & ~present,
             'ID_test': df['split'].eq('test') & ~present,
             'shifted_test': df['split'].eq('test') & present}
    parts = {name: df.loc[mask].copy() for name, mask in masks.items()}
    for name, frame in parts.items():
        if frame.empty or frame.y.nunique() != 2:
            raise ValueError(f'Empty or single-class population: {name}')
    inspection = {'source_rows': raw_n, 'excluded_text_rows': excluded_n,
                  'source_columns': columns, 'toxicity_summary': df.toxicity.describe().to_dict(),
                  'identity_ranges': df[IDENTITIES].agg(['min', 'max']).to_dict()}
    (output / 'data_inspection.json').write_text(json.dumps(inspection, indent=2), encoding='utf-8')
    return parts, population, inspection


def audit_overlap(parts, output):
    frames = {name: frame.assign(normalized_text=normalize_text(frame.comment_text))
              for name, frame in parts.items()}
    within = []
    for name, frame in frames.items():
        within.append(dict(partition=name, n_rows=len(frame),
            missing_comment_ids=int(frame.id.isna().sum()),
            missing_article_ids=int(frame.article_id.isna().sum()),
            rows_in_repeated_id_groups=int(frame.id.duplicated(keep=False).sum()),
            rows_in_repeated_text_groups=int(frame.normalized_text.duplicated(keep=False).sum())))
    rows = []
    for left, right in combinations(frames, 2):
        for key in ['id', 'normalized_text', 'article_id']:
            a, b = frames[left][key], frames[right][key]
            shared = set(a.dropna()) & set(b.dropna())
            rows.append(dict(partition_A=left, partition_B=right, key=key,
                shared_unique_values=len(shared), affected_rows_A=int(a.isin(shared).sum()),
                affected_rows_B=int(b.isin(shared).sum())))
    save_table(pd.DataFrame(within), output, 'within_partition_audit')
    cross = save_table(pd.DataFrame(rows), output, 'cross_partition_overlap')
    if any(row['rows_in_repeated_id_groups'] for row in within):
        raise ValueError('Repeated comment IDs within an analysis partition.')
    if cross.loc[cross.key.eq('id'), 'shared_unique_values'].sum():
        raise ValueError('Comment IDs overlap across analysis partitions.')
    development_test = cross.partition_A.isin(['train', 'validation']) & cross.partition_B.isin(['ID_test', 'shifted_test'])
    if cross.loc[development_test & cross.key.eq('article_id'), 'shared_unique_values'].sum():
        raise ValueError('Development and test article overlap: review the design before fitting.')


def remove_repeated_text(parts, output):
    combined = pd.concat([f.assign(analysis_partition=n) for n, f in parts.items()], ignore_index=True)
    combined['normalized_text'] = normalize_text(combined.comment_text)
    combined['exclude_repeated_text'] = combined.duplicated('normalized_text', keep=False)
    summary = combined.groupby('analysis_partition', sort=False).agg(
        rows_before=('y', 'size'), rows_removed=('exclude_repeated_text', 'sum')).reset_index()
    summary['rows_after'] = summary.rows_before - summary.rows_removed
    save_table(summary, output, 'duplicate_exclusion_summary')
    save_table(combined.loc[combined.exclude_repeated_text,
        ['analysis_partition', 'source_row', 'id', 'article_id']], output, 'excluded_duplicate_records')
    retained = combined.loc[~combined.exclude_repeated_text].copy()
    if retained.normalized_text.duplicated().any():
        raise ValueError('Repeated normalized text remains.')
    cleaned = {n: retained.loc[retained.analysis_partition.eq(n)].drop(
        columns=['analysis_partition', 'normalized_text', 'exclude_repeated_text']).copy() for n in parts}
    for name, frame in cleaned.items():
        if frame.empty or frame.y.nunique() != 2:
            raise ValueError(f'Empty or single-class cleaned partition: {name}')
    save_table(retained[['analysis_partition', 'source_row', 'id', 'article_id', 'y']],
               output, 'analysis_record_manifest')
    final = pd.DataFrame([dict(partition=n, n=len(f), toxic_n=int(f.y.sum()),
        toxicity_prevalence=float(f.y.mean())) for n, f in cleaned.items()])
    save_table(final, output, 'final_partition_summary')
    return cleaned, summary


def fit_model(parts, output, seed):
    model = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True)),
        ('classifier', LogisticRegression(C=1.0, solver='liblinear', max_iter=1000,
                                         class_weight=None, random_state=seed))])
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(parts['train'].comment_text, parts['train'].y)
    joblib.dump(model, output / 'text_model.joblib')
    metadata = dict(training_records=len(parts['train']),
        training_toxicity_prevalence=float(parts['train'].y.mean()),
        vocabulary_size=len(model.named_steps['tfidf'].vocabulary_),
        training_seconds=time.perf_counter() - start,
        solver_iterations=model.named_steps['classifier'].n_iter_.tolist(),
        decision_threshold=THRESHOLD, random_state=seed,
        tfidf_parameters=dict(ngram_range=[1, 2], min_df=3, max_features=50000, sublinear_tf=True),
        classifier_parameters=dict(C=1.0, solver='liblinear', max_iter=1000, class_weight=None))
    (output / 'model_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return model


def safe_ratio(a, b):
    return float(a / b) if b else np.nan


def count_scores(totals):
    tn, fp, fn, tp, brier_sum, log_sum = totals
    n = tn + fp + fn + tp
    recall, specificity = safe_ratio(tp, tp + fn), safe_ratio(tn, tn + fp)
    return dict(accuracy=safe_ratio(tn + tp, n), balanced_accuracy=(recall + specificity) / 2,
        precision=float(tp / (tp + fp)) if tp + fp else 0.0, recall=recall,
        specificity=specificity, f1=float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
        brier_score=safe_ratio(brier_sum, n), log_loss=safe_ratio(log_sum, n))


def contributions(frame):
    y = frame.y.to_numpy(dtype=int)
    h = frame.prediction.to_numpy(dtype=int)
    p = frame.probability.to_numpy(dtype=float)
    eps = np.finfo(float).eps
    clipped = np.clip(p, eps, 1 - eps)
    return np.column_stack([((y == 0) & (h == 0)).astype(float),
        ((y == 0) & (h == 1)).astype(float), ((y == 1) & (h == 0)).astype(float),
        ((y == 1) & (h == 1)).astype(float), (p - y) ** 2,
        -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))])


def summarize_predictions(frame):
    totals = contributions(frame).sum(axis=0)
    tn, fp, fn, tp = totals[:4]
    return dict(n=len(frame), toxic_n=int(tp + fn), non_toxic_n=int(tn + fp),
        toxicity_prevalence=float(frame.y.mean()), **count_scores(totals),
        true_negative=int(tn), false_positive=int(fp), false_negative=int(fn), true_positive=int(tp))


def calibration_bins(predictions):
    data = predictions.assign(bin=np.minimum((predictions.probability * 10).astype(int), 9))
    return data.groupby(['partition', 'bin'], observed=True).agg(
        n=('y', 'size'), mean_predicted_probability=('probability', 'mean'),
        observed_toxicity_fraction=('y', 'mean')).reset_index()


def evaluate_model(model, parts, output):
    rows, frames = [], []
    positive = list(model.named_steps['classifier'].classes_).index(1)
    for name in ENVIRONMENTS:
        f = parts[name]
        saved = f[['source_row', 'id', 'article_id', 'y'] + IDENTITIES].copy()
        saved['partition'] = name
        saved['probability'] = model.predict_proba(f.comment_text)[:, positive]
        saved['prediction'] = (saved.probability >= THRESHOLD).astype(int)
        saved['word_count'] = f.comment_text.str.split().str.len().to_numpy()
        bins = calibration_bins(saved)
        ece = float((bins.n / len(saved) * abs(bins.mean_predicted_probability - bins.observed_toxicity_fraction)).sum())
        rows.append(dict(partition=name, **summarize_predictions(saved),
            roc_auc=roc_auc_score(saved.y, saved.probability),
            average_precision=average_precision_score(saved.y, saved.probability), ece_10_bins=ece))
        frames.append(saved)
    predictions = pd.concat(frames, ignore_index=True)
    metrics = pd.DataFrame(rows)
    indexed = metrics.set_index('partition')
    comparison = pd.DataFrame(dict(metric=SCORES, ID=indexed.loc['ID_test', SCORES].to_numpy(dtype=float),
        shifted=indexed.loc['shifted_test', SCORES].to_numpy(dtype=float)))
    comparison['change_shifted_minus_ID'] = comparison.shifted - comparison.ID
    comparison['better_direction'] = ['lower' if k in LOWER_BETTER else 'higher' for k in SCORES]
    for name, data in [('evaluation_predictions', predictions), ('evaluation_metrics', metrics),
                       ('environment_comparison', comparison)]:
        save_table(data, output, name)
    return predictions, metrics, comparison


def bootstrap_environment_changes(predictions, output, repetitions, seed):
    test = predictions.loc[predictions.partition.isin(['ID_test', 'shifted_test'])].copy()
    codes, articles = pd.factorize(test.article_id, sort=True)
    test['article_code'] = codes
    count = len(articles)
    arrays = {}
    for name in ['ID_test', 'shifted_test']:
        f = test.loc[test.partition.eq(name)]
        aggregate = np.zeros((count, 6))
        np.add.at(aggregate, f.article_code.to_numpy(), contributions(f))
        arrays[name] = aggregate
    observed = {n: count_scores(a.sum(axis=0)) for n, a in arrays.items()}
    rng = np.random.default_rng(seed)
    draws = np.empty((repetitions, len(BOOT_SCORES)))
    for b in range(repetitions):
        selected = rng.integers(0, count, size=count)
        scores = {n: count_scores(a[selected].sum(axis=0)) for n, a in arrays.items()}
        draws[b] = [scores['shifted_test'][k] - scores['ID_test'][k] for k in BOOT_SCORES]
        if (b + 1) % 200 == 0:
            print(f'  Bootstrap {b + 1}/{repetitions}', flush=True)
    if not np.isfinite(draws).all():
        raise ValueError('A bootstrap replicate lacks a required class; review resampling.')
    result = pd.DataFrame(dict(metric=BOOT_SCORES,
        ID=[observed['ID_test'][k] for k in BOOT_SCORES],
        shifted=[observed['shifted_test'][k] for k in BOOT_SCORES],
        change_shifted_minus_ID=[observed['shifted_test'][k] - observed['ID_test'][k] for k in BOOT_SCORES],
        ci_95_low=np.quantile(draws, .025, axis=0), ci_95_high=np.quantile(draws, .975, axis=0),
        valid_repetitions=repetitions))
    result['interval_excludes_zero'] = result.ci_95_low.gt(0) | result.ci_95_high.lt(0)
    save_table(result, output, 'cluster_bootstrap_comparison')
    save_table(pd.DataFrame(draws, columns=BOOT_SCORES), output, 'cluster_bootstrap_draws')
    return result, count


def analyze_subgroups(predictions, length_cutoff, output):
    test = predictions.loc[predictions.partition.isin(['ID_test', 'shifted_test'])].copy()
    shifted = test.loc[test.partition.eq('shifted_test')]
    identity = pd.DataFrame([dict(identity=k, **summarize_predictions(shifted.loc[shifted[k].ge(.5)]))
                             for k in IDENTITIES])
    test['length_group'] = np.where(test.word_count <= length_cutoff, 'short', 'long')
    rows = [dict(partition=n, length_group=g, **summarize_predictions(
        test.loc[test.partition.eq(n) & test.length_group.eq(g)]))
        for n in ['ID_test', 'shifted_test'] for g in ['short', 'long']]
    lengths = pd.DataFrame(rows)
    changes = []
    for group in ['short', 'long']:
        f = lengths.loc[lengths.length_group.eq(group)].set_index('partition')
        for k in ['balanced_accuracy', 'recall', 'specificity']:
            a, b = float(f.loc['ID_test', k]), float(f.loc['shifted_test', k])
            changes.append(dict(length_group=group, metric=k, ID=a, shifted=b, change_shifted_minus_ID=b-a))
    for name, data in [('identity_subgroup_metrics', identity), ('length_subgroup_metrics', lengths),
                       ('length_subgroup_comparison', pd.DataFrame(changes))]:
        save_table(data, output, name)
    (output / 'subgroup_definitions.json').write_text(json.dumps(dict(length_cutoff_words=length_cutoff,
        short_definition='word_count <= cutoff', long_definition='word_count > cutoff',
        cutoff_source='cleaned training partition', identity_threshold=.5), indent=2), encoding='utf-8')
    return identity, lengths


def diagnose_shift(model, parts, predictions, length_cutoff, output):
    vectorizer = model.named_steps['tfidf']
    analyzer, vocabulary = vectorizer.build_analyzer(), vectorizer.vocabulary_
    rows = []
    for name in ['ID_test', 'shifted_test']:
        f = predictions.loc[predictions.partition.eq(name)]
        outside = total = zero = 0
        for text in parts[name].comment_text:
            terms = analyzer(text)
            outside += sum(t not in vocabulary for t in terms)
            total += len(terms)
            zero += not any(t in vocabulary for t in terms)
        scores = summarize_predictions(f)
        rows.append(dict(partition=name, n=len(f), observed_toxicity_prevalence=float(f.y.mean()),
            mean_predicted_toxicity=float(f.probability.mean()),
            mean_probability_minus_prevalence=float(f.probability.mean()-f.y.mean()),
            mean_words=float(f.word_count.mean()), median_words=float(f.word_count.median()),
            fraction_short=float((f.word_count <= length_cutoff).mean()),
            outside_vocabulary_ngram_fraction=safe_ratio(outside, total),
            zero_feature_comment_fraction=zero / len(f),
            **{k: scores[k] for k in ['accuracy', 'recall', 'specificity']}))
    diagnostics = save_table(pd.DataFrame(rows), output, 'shift_diagnostics')
    indexed = diagnostics.set_index('partition')
    pi = indexed.loc['ID_test', 'observed_toxicity_prevalence']
    shifted = indexed.loc['shifted_test']
    standardized = pi * shifted.recall + (1-pi) * shifted.specificity
    standardization = pd.DataFrame([dict(ID_accuracy=indexed.loc['ID_test', 'accuracy'],
        raw_shifted_accuracy=shifted.accuracy, shifted_accuracy_at_ID_prevalence=standardized,
        raw_accuracy_gap_shifted_minus_ID=shifted.accuracy-indexed.loc['ID_test', 'accuracy'],
        standardized_gap_shifted_minus_ID=standardized-indexed.loc['ID_test', 'accuracy'])])
    save_table(standardization, output, 'prevalence_standardization')
    bins = calibration_bins(predictions.loc[predictions.partition.isin(['ID_test', 'shifted_test'])])
    save_table(bins, output, 'calibration_bins')
    return diagnostics, bins


def create_figures(output):
    """Create report figures from results; callable independently by Quarto."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(output / f'{name}.png', dpi=200)
        plt.close(fig)
    bins = pd.read_csv(output / 'calibration_bins.csv')
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in ['ID_test', 'shifted_test']:
        f = bins.loc[bins.partition.eq(name)]
        ax.plot(f.mean_predicted_probability, f.observed_toxicity_fraction, 'o-', label=name)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.set(xlabel='Mean predicted toxicity probability', ylabel='Observed toxicity fraction',
           title='Fixed-model reliability diagram', xlim=(0, 1), ylim=(0, 1))
    ax.legend(); finish(fig, 'reliability_diagram')
    comparison = pd.read_csv(output / 'environment_comparison.csv').set_index('metric')
    keys = ['balanced_accuracy', 'recall', 'specificity', 'f1', 'roc_auc']
    f = comparison.loc[keys]
    fig, ax = plt.subplots(figsize=(8, 4.5)); x = np.arange(len(keys))
    ax.bar(x-.18, f.ID, .36, label='ID'); ax.bar(x+.18, f.shifted, .36, label='Shifted')
    ax.set_xticks(x, [k.replace('_', ' ') for k in keys], rotation=15)
    ax.set(ylabel='Score', ylim=(0, 1), title='Same model, different evaluation populations')
    ax.legend(); finish(fig, 'performance_comparison')
    f = pd.read_csv(output / 'identity_subgroup_metrics.csv')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(f.identity, f.recall); ax.set(xlabel='Toxic recall', xlim=(0, 1),
        title='Shifted identity-mention groups (overlapping)')
    finish(fig, 'identity_recall')
    f = pd.read_csv(output / 'length_subgroup_metrics.csv')
    fig, ax = plt.subplots(figsize=(6, 4.5)); x = np.arange(2)
    for offset, name in [(-.18, 'ID_test'), (.18, 'shifted_test')]:
        q = f.loc[f.partition.eq(name)].set_index('length_group').loc[['short', 'long']]
        ax.bar(x+offset, q.balanced_accuracy, .36, label=name)
    ax.set_xticks(x, ['Short', 'Long']); ax.set(ylabel='Balanced accuracy', ylim=(0, 1),
        title='Performance by training-defined length group'); ax.legend()
    finish(fig, 'length_comparison')


def run_analysis(data_path, output_dir=None, repetitions=1000, seed=42):
    """Run all stages fresh and return the principal results."""
    path = Path(data_path).expanduser().resolve()
    output = Path(output_dir or Path(__file__).resolve().parent / 'results').expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Dataset not found: {path}')
    if repetitions < 2:
        raise ValueError('At least two bootstrap repetitions are required.')
    output.mkdir(parents=True, exist_ok=True)
    # Absence of the completion marker identifies an interrupted/failed run.
    completion = output / 'run_metadata.json'
    completion.unlink(missing_ok=True)
    started = time.perf_counter()
    data_checksum = file_hash(path)
    print('[1/8] Load, inspect and prepare populations', flush=True)
    parts, population, inspection = prepare_data(path, output)
    print('[2/8] Audit IDs, articles and repeated text', flush=True)
    audit_overlap(parts, output)
    print('[3/8] Exclude repeated normalized text', flush=True)
    parts, exclusions = remove_repeated_text(parts, output)
    print('[4/8] Fit the fixed model on training records only', flush=True)
    model = fit_model(parts, output, seed)
    print('[5/8] Evaluate validation, ID and shifted populations', flush=True)
    predictions, metrics, comparison = evaluate_model(model, parts, output)
    print('[6/8] Joint article-cluster bootstrap', flush=True)
    bootstrap, article_count = bootstrap_environment_changes(predictions, output, repetitions, seed)
    cutoff = float(parts['train'].comment_text.str.split().str.len().median())
    print('[7/8] Subgroups and descriptive shift diagnostics', flush=True)
    identity, lengths = analyze_subgroups(predictions, cutoff, output)
    diagnostics, bins = diagnose_shift(model, parts, predictions, cutoff, output)
    print('[8/8] Figures and reproducibility metadata', flush=True)
    create_figures(output)
    if file_hash(path) != data_checksum:
        raise RuntimeError('Source CSV changed during execution.')
    metadata = dict(status='complete', experiment='Custom identity-mention population shift; not official WILDS',
        dataset_provenance='Locally processed Kaggle Civil Comments/Jigsaw data; official-release equivalence unverified',
        source_data=str(path), data_sha256=data_checksum, script_sha256=file_hash(__file__),
        python_version=platform.python_version(), packages={k: version(k) for k in
        ['numpy', 'pandas', 'scikit-learn', 'joblib', 'matplotlib']}, seed=seed,
        bootstrap_repetitions=repetitions, joint_test_articles=article_count,
        bootstrap_unit='article_id; joint across ID and shifted tests',
        threshold=THRESHOLD, length_cutoff_words=cutoff,
        duplicate_rule='Exclude all repeated lowercase/whitespace-normalized text among the four analysis partitions',
        validation_role='Diagnostic only; no tuning or threshold selection',
        elapsed_seconds=time.perf_counter()-started,
        outputs=sorted(p.name for p in output.iterdir() if p.is_file() and p.name != completion.name))
    completion.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('\nID VERSUS SHIFTED COMPARISON\n' + comparison.round(4).to_string(index=False))
    print('\nCLUSTER-BOOTSTRAP COMPARISON\n' + bootstrap.round(4).to_string(index=False))
    print(f'\nANALYSIS COMPLETE. Results: {output}', flush=True)
    return dict(metrics=metrics, comparison=comparison, bootstrap=bootstrap, identity=identity,
                lengths=lengths, diagnostics=diagnostics, calibration=bins, metadata=metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Path to processed CSV')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'results')
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run_analysis(args.data, args.output, args.bootstrap, args.seed)


if __name__ == '__main__':
    main()
