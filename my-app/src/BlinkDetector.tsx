import { useState, useEffect, useRef, useCallback } from "react";
import type { Device, StatusResponse} from "./types"
import "./App.css";

const BLINK_THRESHOLD = 10;

// Connect websocket
const useConnectWS = (setTemp: (data:any) => void ) => {
  const ws = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  const connect = useCallback(() => {
    if (ws.current?.readyState == WebSocket.OPEN) return //already open

    ws.current = new WebSocket('ws://localhost:8000/ws/live')

    ws.current.onopen = () => setIsConnected(true);

    ws.current.onmessage = (event) => {
      const data: any = JSON.parse(event.data);
      console.log(data?.data)
      setTemp(data?.data);
    };
    
    ws.current.onclose = () => setIsConnected(false);
  }, [setTemp])

  const disconnect = useCallback(() => {
    ws.current?.close()
  }, [])


  return { connect, disconnect, isConnected }; 
};


export default function BlinkDetector() {
  const [blinks, setBlinks] = useState<number[]>([]);
  const [alert, setAlert] = useState(false);

  const [devices, setDevices] = useState<Device[]>([]);

  const [temp, setTemp] = useState<StatusResponse | null>(null);

  const { connect, disconnect,  isConnected } = useConnectWS(setTemp)


  // Fetch devices on start
  useEffect(() => {
    fetch('http://localhost:8000/api/devices').then(res => res.json()).then(setDevices);
  }, []);

  // API Command Wrappers

  const startSession = async () => {
    await fetch('http://localhost:8000/api/session/start', { method: 'POST' });
    connect()
  };

  const stopSession = async () => {
    await fetch('http://localhost:8000/api/session/stop', { method: 'POST' });
    disconnect()
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

    return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-[#13131a]">
      <h1 className="blinking text-6xl font-bold mb-6 text-gray-100 text-center">i-Blink</h1>
      <div className="bg-[#181820] rounded-xl shadow-lg px-10 py-12 flex flex-col items-center w-full max-w-md">
        <p className="text-white text-lg"> {isConnected ? "Live" : "Offline"}</p>
        {/* <p className="text-white text-lg"> {devices ? JSON.stringify(devices) : ""}</p> */}
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

        <button 
          className="mb-6 px-8 py-3 bg-teal-500 hover:bg-teal-600 text-white text-sm font-semibold rounded transition"
          onClick={() => {
            stopSession
            setTemp(null)
          }
          }>
          Reset
        </button>
      </div>
    </div>
  );
}


