
import { useState } from "react";


const BLINK_WINDOW = 5000; // 5 seconds
const BLINK_THRESHOLD = 10;

export default function BlinkDetector() {

  const [blinks, setBlinks] = useState<number[]>([]);
  const [alert, setAlert] = useState(false);

  const handleBlink = () => {
    const now = Date.now();
    setBlinks(prev => {
      const updated = [...prev.filter(ts => now - ts < BLINK_WINDOW), now];
      if (updated.length >= BLINK_THRESHOLD) setAlert(true);
      return updated;
    });
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-[#F5F5DC] font-['Roboto','Arial','sans-serif']">
      <h1 className="text-3xl font-bold mb-8 text-black text-center font-['Roboto','Arial','sans-serif']">Bank Robbery Alert</h1>
      <button
        className="mb-8 px-8 py-3 bg-green-600 hover:bg-green-700 text-white text-lg font-normal rounded-full transition shadow-lg min-w-[180px] tracking-wide font-['Roboto','Arial','sans-serif']"
        onClick={handleBlink}
      >
        Button
      </button>
      <div className="text-2xl mb-6 text-black font-['Roboto','Arial','sans-serif'] flex items-center gap-2">
        <span className="">Blinks in last <span className="font-semibold">5s</span>:</span>
        <span className="font-bold text-3xl">{blinks.length}</span>
      </div>
      {alert && (
        <div className="flex flex-col gap-2 p-6 rounded-xl bg-[#f8d7da] border border-[#f5c2c7] text-[#842029] w-full max-w-lg shadow-xl items-center">
          <div className="flex items-center mb-1">
            <svg className="mr-3" width="24" height="24" fill="none" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" fill="#f5c2c7"/><path d="M12 8v6" stroke="#842029" strokeWidth="2" strokeLinecap="round"/><circle cx="12" cy="17" r="1.5" fill="#842029"/></svg>
            <span className="font-bold text-2xl">Alert!</span>
          </div>
          <div className="text-lg font-semibold text-center">Abnormal blink pattern detected!</div>
        </div>
      )}
    </div>
  );
}
