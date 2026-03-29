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
    <div className="flex items-center justify-center min-h-screen bg-[#13131a]">
      <div className="bg-[#181820] rounded-xl shadow-lg px-10 py-12 flex flex-col items-center w-full max-w-md">
        <h1 className="text-3xl font-bold mb-6 text-gray-100 text-center">Bank Robbery Alert</h1>
        <button
          className="mb-6 px-8 py-3 bg-green-500 hover:bg-green-600 text-white text-lg font-semibold rounded transition"
          onClick={handleBlink}
        >
          Simulate Blink
        </button>
        <div className="text-lg mb-4 text-gray-300">
          Blinks in last 5s: <span className="font-bold text-white">{blinks.length}</span>
        </div>
        {alert && (
          <div className="p-4 bg-red-600 text-white rounded font-semibold text-center w-full shadow-md">
            Alert! Abnormal blink pattern detected!
          </div>
        )}
      </div>
    </div>
  );
}
