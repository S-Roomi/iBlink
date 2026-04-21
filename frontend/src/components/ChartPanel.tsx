type Props = {
  alertTriggered: boolean;
};

export default function ChartPanel({ alertTriggered }: Props) {
  return (
    <section className="chartpanel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Instructions</p>
          <h2 className="section-title">What to do next</h2>
        </div>
      </div>

      <div className={`chart-frame chart-frame--instructions ${alertTriggered ? "chart-frame--alert" : ""}`}>
        <div className="instruction-card">
          <p className="instruction-card__lead">Take a breath. You are not alone.</p>
          <p className="instruction-card__body">
            If you are reading this, stay calm. Do not panic. Do not draw attention to the screen.
            Act naturally and keep your face relaxed.
          </p>

          <div className="instruction-card__steps">
            <p>1. Press <strong>Start detection</strong>.</p>
            <p>2. Sit still for 3 seconds while the system sets itself up.</p>
            <p>3. Look naturally at the screen and breathe normally.</p>
            <p>4. Once the status shows <strong>MONITORING</strong>, the system is watching.</p>
            <p>5. Blink 5 times, one per second, naturally and firmly.</p>
          </div>

          <p className="instruction-card__closing">That is it. Help is on the way.</p>
        </div>
      </div>
    </section>
  );
}
