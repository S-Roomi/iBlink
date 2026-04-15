type Props = {
  chartSrc: string;
  onRefresh: () => void;
  hasSession: boolean;
  alertTriggered: boolean;
};

export default function ChartPanel({ chartSrc, onRefresh, hasSession, alertTriggered }: Props) {
  return (
    <section className="chartpanel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Diagnostics</p>
          <h2 className="section-title">Signal chart</h2>
        </div>
        <button onClick={onRefresh} className="button button--ghost">
          Refresh chart
        </button>
      </div>

      <div className={`chart-frame ${alertTriggered ? "chart-frame--alert" : ""}`}>
        {hasSession ? (
          <img src={chartSrc} alt="Session chart" className="chartpanel-image" />
        ) : (
          <div className="chart-empty">
            <p className="chart-empty__title">No chart yet</p>
          </div>
        )}
      </div>
    </section>
  );
}
