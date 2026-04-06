import React, { useState, useEffect, useRef } from "react";
import type { SessionStatus, Device, StatusResponse} from "./types"
import "./App.css";

const BLINK_THRESHOLD = 10;

export default function BlinkDetector() {
  const [blinks, setBlinks] = useState<number[]>([]);
  const [alert, setAlert] = useState(false);

  const [devices, setDevices] = useState<Device[]>([]);
  //const [status, setStatus] = useState<SessionStatus | null>(null);
  const [isConnected, setIsConnected] = useState<boolean>(false);

  const [temp, setTemp] = useState<StatusResponse | null>(null);

  const ws = useRef<WebSocket | null>(null);

  // Fetch all devices on mount
  useEffect(() => {
    fetch('http://localhost:8000/api/devices').then(res => res.json()).then(setDevices);
  }, []);

  // Connect websocket
  const connectWS = () => {

    useEffect(() => {
      ws.current = new WebSocket('ws://localhost:8000/ws/live'); 
      ws.current.onopen = () => setIsConnected(true);
      ws.current.onmessage = (event) => {
        const data: StatusResponse = JSON.parse(event.data);
        setTemp(data);
      }
      ws.current.onclose = () => setIsConnected(false);
    }, [])
    
  }

  // API Command Wrappers
  const startSession = async () => {
    await fetch('http://localhost:8000/api/session/start', { method: 'POST' });
    connectWS(); // Connect WS when session starts
  };

  const stopSession = async () => {
    await fetch('http://localhost:8000/api/session/stop', { method: 'POST' });
    
    //check to see if ws is open. If so close it
    ws.current?.close();
  };

  const dismissAlert = () => {
    fetch('http://localhost:8000/api/session/dismiss', { method: 'POST' });
  };

  const get_status = async () => {
    await fetch('http://localhost:8000/api/session/status', { method: 'GET'})
    .then(res => res.json())
    .then(setTemp)
    console.log(temp)
    
  }
  
  const chart = () => {
    fetch('http://localhost:8000/api/session/chart', { method: 'GET'})
  }

  const handleBlink = () => {
    const now = Date.now();
    setBlinks(prev => {
      const updated = [...prev, now];
      if (updated.length >= BLINK_THRESHOLD) setAlert(true);
      return updated;
    });
  };
  const resetBlink = () => {
    setBlinks([])
    setAlert(false)
  }


    return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-[#13131a]">
      <h1 className="blinking text-6xl font-bold mb-6 text-gray-100 text-center">i-Blink</h1>
      <div className="bg-[#181820] rounded-xl shadow-lg px-10 py-12 flex flex-col items-center w-full max-w-md">
        <button
          className="mb-6 px-8 py-3 bg-green-500 hover:bg-green-600 text-white text-md font-semibold rounded transition"
          onClick={startSession}
        >
          Simulate Blink
        </button>

        <button
          className="mb-6 px-8 py-3 bg-red-500 hover:bg-red-600 text-white text-sm font-semibold rounded transition"
          onClick={stopSession}
        >
          Stop Simulation
        </button>

        <button
          className="mb-6 px-8 py-3 bg-teal-500 hover:bg-teal-600 text-white text-sm font-semibold rounded transition"
          onClick={stopSession}
        >
          Dismiss Alert
        </button>

        <div className="text-lg mb-4 text-gray-300">
          Total blinks: <span className="font-bold text-white">{temp?.blink_count ? temp?.blink_count : 0 }</span>
        </div>

        <button
          className="mb-6 px-8 py-3 bg-teal-500 hover:bg-teal-600 text-white text-sm font-semibold rounded transition"
          onClick={get_status}
        >
          Status
        </button>

        <button
          className="mb-6 px-8 py-3 bg-teal-500 hover:bg-teal-600 text-white text-sm font-semibold rounded transition"
          onClick={chart}
        >
          Chart
        </button>
      </div>
    </div>
  );

  // return (
  //   <div className="p-4 border rounded-lg shadow-md">
  //     <h2 className="text-xl font-bold mb-4">Acoustic Blink Detector</h2>
      
  //     <div className="space-x-2 mb-6">
  //       <button onClick={startSession} className="bg-green-500 text-white px-4 py-2 rounded">Start</button>
  //       <button onClick={stopSession} className="bg-red-500 text-white px-4 py-2 rounded">Stop</button>
  //       <button onClick={dismissAlert} className="bg-gray-500 text-white px-4 py-2 rounded">Dismiss Alert</button>
  //     </div>

  //     {/* Visual Feedback */}
  //     <div className={`p-4 rounded ${status?.alert_triggered ? 'bg-red-200 animate-pulse' : 'bg-blue-50'}`}>
  //       <p>Status: {isConnected ? "🟢 Connected" : "🔴 Disconnected"}</p>
  //       <p>Alert State: <strong>{status?.alert_triggered ? "BLINK DETECTED" : "Normal"}</strong></p>
  //     </div>

  //     {/* Real-time Chart Image */}
  //     {isConnected && (
  //       <div className="mt-4">
  //         <h3 className="font-semibold">Live Signal:</h3>
  //         <img 
  //           src={`http://localhost:8000/api/session/chart?t=${Date.now()}`} 
  //           alt="FMCW Signal" 
  //           className="w-full border"
  //         />
  //       </div>
  //     )}
  //   </div>
  // );
}


