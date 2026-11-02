import pandas as pd, pathlib, sys

def main():
    # Paths
    trace_path = pathlib.Path(r'C:\\Users\\varsa\\Desktop\\DriftGuard\\dg11\\results\\v1.1\\guard_stress\\a6_guard_stress_traces.csv.gz')
    out_path = pathlib.Path(r'C:\\Users\\varsa\\Desktop\\DriftGuard\\dg11\\results\\v1.1\\guard_stress\\event_level_analysis.csv')
    if not trace_path.is_file():
        sys.exit(f'Trace file not found: {trace_path}')
    # Load trace
    df = pd.read_csv(trace_path, compression='gzip')
    # Verify expected columns
    expected = {'scenario','attack_fraction_requested','seed','window','guard_warm','event_index','label','score','admitted','tau','window_n','window_attack_n','fallback','future_in_buffer'}
    missing = expected - set(df.columns)
    if missing:
        sys.exit(f'Missing columns in trace CSV: {missing}')
    # Rename columns to more explicit names for the hand‑over
    df = df.rename(columns={
        'event_index':'event_index',
        'tau':'threshold_before',
        'window_n':'model_population',
        'future_in_buffer':'future_in_buffer_flag',
    })
    # Reorder columns for readability
    cols_order = ['scenario','attack_fraction_requested','seed','window','guard_warm','event_index','label','score','threshold_before','admitted','model_population','fallback','future_in_buffer_flag','window_attack_n']
    df = df[cols_order]
    # Write CSV (no index)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f'Event‑level analysis written to {out_path}')

if __name__ == '__main__':
    main()
