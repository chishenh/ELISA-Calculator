import React, { useState, useMemo, useRef, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Scatter, ComposedChart
} from 'recharts';
import { Settings, Edit3, Activity, Calculator, RefreshCw, Trash2, FileText, BarChart2, Download, Grid, Play, Clipboard, RotateCcw, XCircle, AlertTriangle, Sigma, Info } from 'lucide-react';
import ExcelJS from 'exceljs';
import { saveAs } from 'file-saver';

// --- Helper: Dynamic Script Loader for PDF Libraries ---
const loadScript = (src) => {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) {
      resolve();
      return;
    }
    const script = document.createElement('script');
    script.src = src;
    script.onload = resolve;
    script.onerror = reject;
    document.head.appendChild(script);
  });
};

// --- Matrix Math Helpers for LMA ---
const Matrix = {
  transpose: (A) => A[0].map((_, colIndex) => A.map(row => row[colIndex])),
  multiply: (A, B) => {
    const m = A.length, n = A[0].length, p = B[0].length;
    let C = Array(m).fill(0).map(() => Array(p).fill(0));
    for (let i = 0; i < m; i++) for (let j = 0; j < p; j++) for (let k = 0; k < n; k++) C[i][j] += A[i][k] * B[k][j];
    return C;
  },
  multiplyVector: (A, v) => A.map(row => row.reduce((sum, elm, i) => sum + elm * v[i], 0)),
  solveLinearSystem: (A, b) => {
    const n = A.length;
    let M = A.map((row, i) => [...row, b[i]]);
    for (let i = 0; i < n; i++) {
      let maxRow = i;
      for (let k = i + 1; k < n; k++) if (Math.abs(M[k][i]) > Math.abs(M[maxRow][i])) maxRow = k;
      [M[i], M[maxRow]] = [M[maxRow], M[i]];
      for (let k = i + 1; k < n; k++) {
        const factor = M[k][i] / M[i][i];
        for (let j = i; j <= n; j++) M[k][j] -= factor * M[i][j];
      }
    }
    let x = Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
      let sum = 0;
      for (let j = i + 1; j < n; j++) sum += M[i][j] * x[j];
      x[i] = (M[i][n] - sum) / M[i][i];
    }
    return x;
  }
};

// --- 5PL Logic ---
const calculate5PL = (x, [a, d, c, b, g]) => {
  if (x <= 0) return a;
  return d + (a - d) / Math.pow((1 + Math.pow(x / c, b)), g);
};

const calculateConcentration = (y, [a, d, c, b, g]) => {
  const minAsymp = Math.min(a, d);
  const maxAsymp = Math.max(a, d);
  const eps = 1e-6;
  if (y < minAsymp - eps || y > maxAsymp + eps) return NaN;

  try {
    const term1 = (a - d) / (y - d);
    if (term1 <= 0) return NaN;
    const term2 = Math.pow(term1, 1 / g) - 1;
    if (term2 < 0) return NaN;
    return c * Math.pow(term2, 1 / b);
  } catch (e) { return NaN; }
};

const solveLevenbergMarquardt = (dataPoints, initialParamsObj) => {
  let p = [initialParamsObj.a, initialParamsObj.d, initialParamsObj.c, initialParamsObj.b, initialParamsObj.g];
  const maxIter = 200;
  let lambda = 0.01;
  const tolerance = 1e-8;

  const getSSE = (params) => {
    if (params.some(val => val <= 1e-9)) return Infinity;
    return dataPoints.reduce((acc, point) => {
      const diff = point.y - calculate5PL(point.x, params);
      return acc + diff * diff;
    }, 0);
  };

  let currentSSE = getSSE(p);

  for (let iter = 0; iter < maxIter; iter++) {
    let J = [], r = [];
    const [a, d, c, b, g] = p;

    for (let point of dataPoints) {
      const { x, y: y_obs } = point;
      const y_pred = calculate5PL(x, p);
      r.push(y_obs - y_pred);

      let row = [0, 0, 0, 0, 0];
      if (x <= 0) {
        row = [1, 0, 0, 0, 0];
      } else {
        const xc = x / c;
        const xc_b = Math.pow(xc, b);
        const base = 1 + xc_b;
        const denom = Math.pow(base, g);
        row[0] = 1 / denom;
        row[1] = 1 - (1 / denom);
        const dy_dbase = (a - d) * (-g) * Math.pow(base, -g - 1);
        const dbase_dc = -(b / c) * xc_b;
        row[2] = dy_dbase * dbase_dc;
        const dbase_db = xc_b * Math.log(xc);
        row[3] = dy_dbase * dbase_db;
        row[4] = - ((a - d) / denom) * Math.log(base);
      }
      J.push(row);
    }

    const JT = Matrix.transpose(J);
    const JTJ = Matrix.multiply(JT, J);
    const JTr = Matrix.multiplyVector(JT, r);

    let A = JTJ.map((row, i) => row.map((val, j) => i === j ? val * (1 + lambda) : val));

    let delta;
    try { delta = Matrix.solveLinearSystem(A, JTr); }
    catch (e) { lambda *= 10; continue; }

    const p_new = p.map((val, i) => val + delta[i]);
    const newSSE = getSSE(p_new);

    if (isFinite(newSSE) && newSSE < currentSSE) {
      p = p_new;
      lambda /= 10;
      if (Math.abs(currentSSE - newSSE) < tolerance) { currentSSE = newSSE; break; }
      currentSSE = newSSE;
    } else {
      lambda *= 10;
      if (lambda > 1e12) break;
    }
  }
  return { a: p[0], d: p[1], c: p[2], b: p[3], g: p[4], sse: currentSSE };
};

const calcStats = (values) => {
  const n = values.length;
  if (n === 0) return { mean: 0, sd: null, cv: null };
  const mean = values.reduce((a, b) => a + b, 0) / n;
  if (n === 1) return { mean, sd: null, cv: null };
  const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (n - 1);
  const sd = Math.sqrt(variance);
  const cv = mean !== 0 ? (sd / mean) * 100 : 0;
  return { mean, sd, cv };
};

// --- Layout Logic Helpers ---
const createEmptyGrid = (val) => Array(8).fill(null).map(() => Array(12).fill(val));

const getAutoLayout = (settings, standards) => {
  const newLayout = createEmptyGrid({ type: 'EMPTY', id: null });
  const { std: stdR, ctl: ctlR, sample: sampleR } = settings;
  const findSlot = (repeat, currentLayout) => {
    for (let c = 0; c <= 12 - repeat; c++) {
      for (let r = 0; r < 8; r++) {
        let fits = true;
        for (let k = 0; k < repeat; k++) {
          if (currentLayout[r][c + k].type !== 'EMPTY') { fits = false; break; }
        }
        if (fits) return { r, c };
      }
    }
    return null;
  };
  for (let i = 0; i < standards.length; i++) {
    const slot = findSlot(stdR, newLayout);
    if (slot) {
      for (let k = 0; k < stdR; k++) {
        const idStr = (i + 1).toString().padStart(2, '0');
        newLayout[slot.r][slot.c + k] = { type: 'STD', id: idStr };
      }
    }
  }
  const controls = ['H', 'L'];
  for (let ctlId of controls) {
    const slot = findSlot(ctlR, newLayout);
    if (slot) {
      for (let k = 0; k < ctlR; k++) {
        newLayout[slot.r][slot.c + k] = { type: 'CTL', id: ctlId };
      }
    }
  }
  let sampleIdx = 1;
  while (true) {
    const slot = findSlot(sampleR, newLayout);
    if (!slot) break;
    const idStr = `S${sampleIdx.toString().padStart(2, '0')}`;
    for (let k = 0; k < sampleR; k++) {
      newLayout[slot.r][slot.c + k] = { type: 'UNK', id: idStr };
    }
    sampleIdx++;
  }
  return newLayout;
};

const INITIAL_STD_CONCS = [
  { id: 1, conc: 250 }, { id: 2, conc: 125 }, { id: 3, conc: 62.5 }, { id: 4, conc: 31.25 },
  { id: 5, conc: 15.625 }, { id: 6, conc: 7.813 }, { id: 7, conc: 3.906 }, { id: 8, conc: 0 }
];
const INITIAL_REPEAT_SETTINGS = { std: 2, ctl: 2, sample: 1 };

// --- UI Components ---

const PlateGrid = ({ data, setData, type = "layout", activeTool }) => {
  const rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];
  const cols = Array.from({ length: 12 }, (_, i) => i + 1);

  const getCellLabel = (cell) => {
    if (cell.type === 'EMPTY') return '';
    if (cell.type === 'STD') return `STD${cell.id}`;
    if (cell.type === 'CTL') return `CTL-${cell.id}`;
    if (cell.type === 'UNK') return cell.id;
    if (cell.type === 'BLK') return `BLK`;
    if (cell.type === 'UNK' && isNaN(cell.id)) return cell.id;
    return cell.id;
  };

  const handleCellClick = (rIndex, cIndex) => { };

  const handleTextChange = (rIndex, cIndex, val, field) => {
    const newData = [...data];
    if (field === 'id') {
      let newType = 'UNK';
      let newId = val;
      const upperVal = val.toUpperCase();
      if (upperVal.startsWith('STD')) {
        newType = 'STD';
        newId = val.substring(3);
      } else if (upperVal.startsWith('CTL')) {
        newType = 'CTL';
        newId = val.replace(/^CTL-?/i, '');
      } else if (upperVal === 'BLK' || upperVal === 'BLANK') {
        newType = 'BLK';
        newId = '';
      } else if (val === '') {
        newType = 'EMPTY';
        newId = null;
      } else {
        newType = 'UNK';
        newId = val;
      }
      newData[rIndex][cIndex] = { type: newType, id: newId };
    } else {
      newData[rIndex][cIndex] = val;
    }
    setData(newData);
  };

  const handlePaste = (e, rIndex, cIndex, field) => {
    e.preventDefault();
    const clipboardData = e.clipboardData.getData('text');
    const pasteRows = clipboardData.split(/\r\n|\n|\r/).filter(row => row.trim() !== '');
    if (pasteRows.length === 0) return;
    const newData = [...data];
    pasteRows.forEach((rowStr, rOffset) => {
      const targetRow = rIndex + rOffset;
      if (targetRow >= 8) return;
      const pasteCols = rowStr.split('\t');
      pasteCols.forEach((val, cOffset) => {
        const targetCol = cIndex + cOffset;
        if (targetCol >= 12) return;
        const cleanVal = val.trim();
        if (type === 'values') {
          newData[targetRow][targetCol] = cleanVal;
        } else {
          if (field === 'id') {
            let newType = 'UNK';
            let newId = cleanVal;
            const upperVal = cleanVal.toUpperCase();
            if (upperVal.startsWith('STD')) { newType = 'STD'; newId = cleanVal.substring(3); }
            else if (upperVal.startsWith('CTL')) { newType = 'CTL'; newId = cleanVal.replace(/^CTL-?/i, ''); }
            else if (upperVal === 'BLK') { newType = 'BLK'; newId = ''; }
            else if (cleanVal === '') { newType = 'EMPTY'; newId = null; }
            else { newType = 'UNK'; newId = cleanVal; }
            newData[targetRow][targetCol] = { type: newType, id: newId };
          }
        }
      });
    });
    setData(newData);
  };

  return (
    <div className="overflow-x-auto pb-2">
      <div className="grid grid-cols-[30px_repeat(12,minmax(55px,1fr))] gap-1 min-w-[800px]">
        <div className="col-start-1"></div>
        {cols.map(c => <div key={c} className="text-center text-xs font-bold text-gray-500">{c}</div>)}
        {rows.map((row, rIndex) => (
          <React.Fragment key={row}>
            <div className="flex items-center justify-center text-xs font-bold text-gray-500">{row}</div>
            {cols.map((col, cIndex) => {
              const cellData = data[rIndex][cIndex];
              if (type === "layout") {
                let bgColor = "bg-gray-50";
                let borderColor = "border-gray-200";
                let textColor = "text-gray-400";
                if (cellData.type === 'STD') { bgColor = "bg-blue-50"; borderColor = "border-blue-300"; textColor = "text-blue-700"; }
                else if (cellData.type === 'CTL') { bgColor = "bg-purple-50"; borderColor = "border-purple-300"; textColor = "text-purple-700"; }
                else if (cellData.type === 'UNK') { bgColor = "bg-amber-50"; borderColor = "border-amber-300"; textColor = "text-amber-700"; }
                else if (cellData.type === 'BLK') { bgColor = "bg-slate-200"; borderColor = "border-slate-400"; textColor = "text-slate-600"; }
                const displayValue = getCellLabel(cellData);
                return (
                  <div key={`${row}${col}`} className={`aspect-square border ${borderColor} rounded flex items-center justify-center p-1 ${bgColor} relative group`}>
                    <input type="text" className={`w-full h-full text-center text-[10px] font-bold bg-transparent outline-none p-0 ${textColor} placeholder-gray-300`} value={displayValue} onChange={(e) => handleTextChange(rIndex, cIndex, e.target.value, 'id')} onPaste={(e) => handlePaste(e, rIndex, cIndex, 'id')} placeholder="..." />
                  </div>
                );
              } else {
                return (
                  <input key={`${row}${col}`} type="number" step="0.001" className="aspect-square border border-gray-200 rounded text-center text-xs w-full focus:ring-2 focus:ring-indigo-500 outline-none p-0" value={cellData} onChange={(e) => handleTextChange(rIndex, cIndex, e.target.value, 'val')} onPaste={(e) => handlePaste(e, rIndex, cIndex, 'val')} placeholder="-" />
                );
              }
            })}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
};

// Simple confirm button component
const ConfirmButton = ({ onConfirm, label, icon: Icon, className }) => {
  const [confirming, setConfirming] = useState(false);
  useEffect(() => {
    if (confirming) {
      const timer = setTimeout(() => setConfirming(false), 3000);
      return () => clearTimeout(timer);
    }
  }, [confirming]);
  return (
    <button
      onClick={() => { if (confirming) { onConfirm(); setConfirming(false); } else { setConfirming(true); } }}
      className={`${className} transition-all duration-200 ${confirming ? 'bg-red-600 text-white border-red-600 animate-pulse' : ''}`}
    >
      {confirming ? <span className="flex items-center gap-1"><AlertTriangle size={12} /> Confirm?</span> : <span className="flex items-center gap-1">{Icon && <Icon size={12} />} {label}</span>}
    </button>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState('layout');
  const [layout, setLayout] = useState(() => getAutoLayout(INITIAL_REPEAT_SETTINGS, INITIAL_STD_CONCS));
  const [odValues, setOdValues] = useState(createEmptyGrid(''));
  const [stdConcs, setStdConcs] = useState(INITIAL_STD_CONCS);
  const [repeatSettings, setRepeatSettings] = useState(INITIAL_REPEAT_SETTINGS);
  const [activeTool, setActiveTool] = useState({ type: 'STD', id: 1 });
  const [fitResult, setFitResult] = useState(null);
  const [calculatedData, setCalculatedData] = useState([]);
  const [chartScale, setChartScale] = useState('log');
  const [isGeneratingDoc, setIsGeneratingDoc] = useState(false);
  const [isGeneratingExcel, setIsGeneratingExcel] = useState(false);
  const chartRef = useRef(null);

  const clearLayout = () => {
    setLayout(createEmptyGrid({ type: 'EMPTY', id: null }));
    setFitResult(null);
    setCalculatedData([]);
  };

  const clearODs = () => {
    setOdValues(createEmptyGrid(''));
    setFitResult(null);
    setCalculatedData([]);
  };

  const clearStandards = () => {
    setStdConcs(stdConcs.map(s => ({ ...s, conc: 0 })));
    setFitResult(null);
    setCalculatedData([]);
  };

  const handleStdPaste = (e, index) => {
    e.preventDefault();
    const clipboardData = e.clipboardData.getData('text');
    const rows = clipboardData.split(/\r\n|\n|\r/).filter(r => r.trim() !== '');
    if (rows.length === 0) return;
    const newStds = [...stdConcs];
    rows.forEach((val, i) => {
      const targetIdx = index + i;
      if (targetIdx < newStds.length) newStds[targetIdx].conc = parseFloat(val) || 0;
    });
    setStdConcs(newStds);
  };

  const applyAutoLayout = () => {
    const newLayout = getAutoLayout(repeatSettings, stdConcs);
    setLayout(newLayout);
    setFitResult(null);
    setCalculatedData([]);
  };

  const loadDemoData = () => {
    setStdConcs(INITIAL_STD_CONCS);
    const demoLayout = getAutoLayout(repeatSettings, INITIAL_STD_CONCS);
    setLayout(demoLayout);
    const newOds = createEmptyGrid('');
    const trueParams = { a: 0.05, d: 2.5, c: 15, b: 1.2, g: 0.8 };
    const p_vec = [trueParams.a, trueParams.d, trueParams.c, trueParams.b, trueParams.g];
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 12; c++) {
        const cell = demoLayout[r][c];
        if (cell.type === 'STD') {
          const stdId = parseInt(cell.id);
          const std = INITIAL_STD_CONCS.find(s => s.id === stdId);
          if (std) {
            const val = calculate5PL(std.conc, p_vec);
            newOds[r][c] = (val + (Math.random() - 0.5) * 0.04).toFixed(3);
          }
        } else if (cell.type === 'CTL') {
          const base = cell.id === 'L' ? 0.2 : 1.8;
          newOds[r][c] = (base + (Math.random() - 0.5) * 0.05).toFixed(3);
        } else if (cell.type === 'UNK') {
          const randConc = Math.random() * 90 + 2;
          const val = calculate5PL(randConc, p_vec);
          newOds[r][c] = (val + (Math.random() - 0.5) * 0.08).toFixed(3);
        }
      }
    }
    setOdValues(newOds);
    setFitResult(null);
    setCalculatedData([]);
  };

  const handleCalculate = () => {
    let points = [];
    for (let r = 0; r < 8; r++) for (let c = 0; c < 12; c++) {
      const cell = layout[r][c];
      const od = parseFloat(odValues[r][c]);
      if (!isNaN(od) && cell.type === 'STD') {
        const idNum = parseInt(cell.id);
        const stdInfo = stdConcs.find(s => s.id === idNum);
        if (stdInfo) points.push({ x: parseFloat(stdInfo.conc), y: od });
      }
    }

    if (points.length < 5) return alert("Please define at least 5 standard points.");

    points.sort((a, b) => a.x - b.x);
    const yStart = points[0].y;
    const yEnd = points[points.length - 1].y;
    const isIncreasing = yEnd > yStart;
    const minY = Math.min(...points.map(p => p.y));
    const maxY = Math.max(...points.map(p => p.y));
    const safeMin = minY > 0 ? minY : 0.001;
    const safeMax = maxY > 0 ? maxY : 0.001;
    let initA = isIncreasing ? safeMin : safeMax;
    let initD = isIncreasing ? safeMax : safeMin;
    const midY = (minY + maxY) / 2;
    let closestP = points[0];
    let minDiff = Math.abs(points[0].y - midY);
    for (let p of points) { const diff = Math.abs(p.y - midY); if (diff < minDiff) { minDiff = diff; closestP = p; } }
    const initC = closestP.x > 0 ? closestP.x : 10;
    const initialParams = { a: initA, d: initD, c: initC, b: 1.0, g: 1.0 };
    const resultParams = solveLevenbergMarquardt(points, initialParams);
    const yMean = points.reduce((acc, p) => acc + p.y, 0) / points.length;
    const ssTot = points.reduce((acc, p) => acc + Math.pow(p.y - yMean, 2), 0);
    const rSquared = ssTot === 0 ? 0 : (1 - (resultParams.sse / ssTot));
    setFitResult({ params: resultParams, rSquared, points });

    let calcResults = [];
    const p_vec = [resultParams.a, resultParams.d, resultParams.c, resultParams.b, resultParams.g];
    for (let r = 0; r < 8; r++) for (let c = 0; c < 12; c++) {
      const cell = layout[r][c];
      const od = parseFloat(odValues[r][c]);
      if (!isNaN(od) && cell.type !== 'EMPTY') {
        const conc = calculateConcentration(od, p_vec);
        const wellName = `${['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'][r]}${c + 1}`;
        const wellIdx = r * 12 + c;
        calcResults.push({ well: wellName, wellIdx, type: cell.type, id: cell.id, od: od, conc: conc, note: isNaN(conc) ? 'Out of Range' : '' });
      }
    }
    setCalculatedData(calcResults);
    setActiveTab('results');
  };

  const getChartData = useMemo(() => {
    if (!fitResult) return { curve: [], scatter: [], xDomain: [0, 100] };
    const { params, points } = fitResult;
    const p_vec = [params.a, params.d, params.c, params.b, params.g];
    let curve = [];
    const minX = 0.1; const maxX = 1000;
    const logMin = Math.log10(minX); const logMax = Math.log10(maxX);
    const steps = 100;
    for (let i = 0; i <= steps; i++) {
      const logX = logMin + (i / steps) * (logMax - logMin);
      const x = Math.pow(10, logX);
      curve.push({ x, y: calculate5PL(x, p_vec) });
    }
    return { curve, scatter: chartScale === 'log' ? points.filter(p => p.x > 0) : points };
  }, [fitResult, chartScale]);

  const handleDownloadPdf = async () => {
    if (!fitResult || calculatedData.length === 0) return alert("Please calculate fit first.");
    setIsGeneratingDoc(true);
    try {
      await loadScript("https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js");
      await loadScript("https://cdnjs.cloudflare.com/ajax/libs/jspdf-autotable/3.5.29/jspdf.plugin.autotable.min.js");
      const { jsPDF } = window.jspdf;
      const doc = new jsPDF();
      doc.setFontSize(18); doc.text("ELISA Analysis Report", 105, 15, null, null, "center");
      doc.setFontSize(10); doc.text(`Generated: ${new Date().toLocaleString()}`, 200, 25, null, null, "right");

      doc.setFontSize(14); doc.text("1. Plate Layout", 14, 30);
      const gridHead = [['', ...Array.from({ length: 12 }, (_, i) => i + 1)]];
      const layoutBody = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'].map((r, i) => [r, ...layout[i].map(c => {
        if (c.type === 'EMPTY') return '';
        if (c.type === 'STD') return `STD${c.id}`;
        if (c.type === 'CTL') return `CTL-${c.id}`;
        if (c.type === 'UNK') return c.id;
        return c.id;
      })]);
      doc.autoTable({ startY: 35, head: gridHead, body: layoutBody, theme: 'grid', styles: { fontSize: 7, halign: 'center', valign: 'middle', minCellHeight: 10 }, headStyles: { fillColor: [100, 100, 100] }, columnStyles: { 0: { fontStyle: 'bold', fillColor: [240, 240, 240] } }, tableWidth: '100%' });

      const nextY = doc.lastAutoTable.finalY + 15;
      doc.setFontSize(14); doc.text("2. OD Value Layout", 14, nextY);
      const odBody = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'].map((r, i) => [r, ...odValues[i].map(val => val === '' ? '-' : val)]);
      doc.autoTable({ startY: nextY + 5, head: gridHead, body: odBody, theme: 'grid', styles: { fontSize: 7, halign: 'center', valign: 'middle', minCellHeight: 8 }, headStyles: { fillColor: [100, 100, 100] }, columnStyles: { 0: { fontStyle: 'bold', fillColor: [240, 240, 240] } }, tableWidth: '100%' });

      doc.addPage();
      doc.setFontSize(16); doc.text("3. Curve Fitting Results", 14, 20);

      let chartY = 30;
      try {
        await loadScript("https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js");
        const chartContainer = document.getElementById('chart-container');
        if (chartContainer) {
          // Wait a moment for any animations to finish (though we disabled them)
          await new Promise(r => setTimeout(r, 100));

          const canvas = await window.html2canvas(chartContainer, {
            scale: 2, // Higher scale for better resolution in PDF
            useCORS: true,
            logging: false,
            backgroundColor: '#ffffff'
          });

          const imgData = canvas.toDataURL('image/png');
          doc.setFontSize(12); doc.text("3.1. Standard Curve Graph", 14, chartY);

          // Calculate dimensions to fit PDF page width (approx 180mm usable width)
          const imgWidth = 180;
          const imgHeight = (canvas.height * imgWidth) / canvas.width;

          doc.addImage(imgData, 'PNG', 15, chartY + 5, imgWidth, imgHeight);
          chartY += imgHeight + 20; // Update Y position for next elements
        }
      } catch (e) { console.error("Graph export error:", e); }

      doc.setFontSize(10);
      doc.text("Model Formula: y = D + (A - D) / ((1 + (x/C)^B)^G)", 14, chartY + 5);

      const { a, d } = fitResult.params;
      const isIncreasing = a < d;
      const descA = isIncreasing ? "Min Asymptote (Bottom)" : "Max Asymptote (Top)";
      const descD = isIncreasing ? "Max Asymptote (Top)" : "Min Asymptote (Bottom)";

      const ec50 = fitResult.params.c * Math.pow((Math.pow(2, 1 / fitResult.params.g) - 1), 1 / fitResult.params.b);
      doc.setFontSize(12); doc.text("3.2. 5PL Parameters & Goodness of Fit", 14, chartY + 12);
      doc.autoTable({
        startY: chartY + 17,
        head: [['Parameter', 'Value', 'Description']],
        body: [
          ["R-Squared", fitResult.rSquared.toFixed(5), "Coefficient of Determination"],
          ["A", fitResult.params.a.toFixed(4), descA],
          ["D", fitResult.params.d.toFixed(4), descD],
          ["C", fitResult.params.c.toFixed(4), "Inflection Point"],
          ["B", fitResult.params.b.toFixed(4), "Hill Slope"],
          ["G", fitResult.params.g.toFixed(4), "Symmetry Factor"],
          ["EC50", ec50.toFixed(4), "Half Maximal Effective Concentration"]
        ],
        theme: 'grid',
        headStyles: { fillColor: [79, 70, 229], halign: 'center' },
        styles: { halign: 'center' },
        tableWidth: '100%'
      });

      doc.addPage();
      doc.setFontSize(16); doc.text("4. Standard Curve Statistics", 14, 20);
      const stdGroups = {};
      calculatedData.filter(d => d.type === 'STD').forEach(d => { if (!stdGroups[d.id]) stdGroups[d.id] = { wells: [] }; stdGroups[d.id].wells.push(d); });
      const stdStatsBody = [];
      Object.keys(stdGroups).sort((a, b) => parseInt(a) - parseInt(b)).forEach(id => {
        const group = stdGroups[id];
        const validConcs = group.wells.map(w => w.conc).filter(c => !isNaN(c));
        const { mean, sd, cv } = calcStats(validConcs);
        const stdDef = stdConcs.find(s => s.id === parseInt(id));
        const theoConc = stdDef ? stdDef.conc : '-';
        group.wells.forEach((well, idx) => {
          stdStatsBody.push([`STD ${id}`, well.well, idx === 0 ? theoConc : '', well.od.toFixed(3), isNaN(well.conc) ? "Out" : well.conc.toFixed(3), idx === 0 ? mean.toFixed(3) : '', idx === 0 ? (sd !== null ? sd.toFixed(3) : 'NA') : '', idx === 0 ? (cv !== null ? cv.toFixed(1) : 'NA') : '']);
        });
      });
      doc.autoTable({ startY: 25, head: [['ID', 'Well', 'Std Conc.', 'OD', 'Back-calc', 'Avg Conc.', 'SD', 'CV (%)']], body: stdStatsBody, theme: 'grid', headStyles: { fillColor: [79, 70, 229], halign: 'center' }, styles: { halign: 'center' }, tableWidth: '100%' });

      doc.addPage();
      doc.setFontSize(16); doc.text("5. Sample Concentrations", 14, 20);
      const sampleGroups = {};
      calculatedData.filter(d => d.type !== 'STD').forEach(d => {
        const key = `${d.type}-${d.id}`;
        if (!sampleGroups[key]) sampleGroups[key] = { type: d.type, id: d.id, wells: [] };
        sampleGroups[key].wells.push(d);
      });
      const sampleBody = [];
      const typePriority = { 'CTL': 0, 'UNK': 1, 'BLK': 2 };
      const getPriority = (t) => typePriority[t] ?? 9;

      Object.values(sampleGroups).sort((a, b) => {
        const pA = getPriority(a.type);
        const pB = getPriority(b.type);
        if (pA !== pB) return pA - pB;
        if (a.type === 'CTL' && b.type === 'CTL') {
          if (a.id === 'H' && b.id === 'L') return -1;
          if (a.id === 'L' && b.id === 'H') return 1;
        }
        return String(a.id).localeCompare(String(b.id), undefined, { numeric: true });
      }).forEach(group => {
        const validConcs = group.wells.map(w => w.conc).filter(c => !isNaN(c));
        const { mean, sd, cv } = calcStats(validConcs);
        const displayName = group.type === 'UNK' ? group.id : group.type === 'CTL' ? `CTL-${group.id}` : group.id;

        group.wells.forEach((well, idx) => {
          sampleBody.push([
            well.well,
            displayName,
            well.od.toFixed(3),
            isNaN(well.conc) ? "Out of Range" : well.conc.toFixed(3),
            idx === 0 ? (isNaN(mean) ? '-' : mean.toFixed(3)) : '',
            idx === 0 ? (sd === null ? '' : sd.toFixed(3)) : '',
            idx === 0 ? (cv === null ? '' : cv.toFixed(1)) : ''
          ]);
        });
      });

      doc.autoTable({ startY: 25, head: [['Well', 'Sample ID', 'OD', 'Concentration', 'Avg Conc.', 'SD', 'CV (%)']], body: sampleBody, theme: 'grid', headStyles: { fillColor: [234, 88, 12], halign: 'center' }, styles: { halign: 'center' }, tableWidth: '100%' });
      
      const today = new Date();
      const dateStr = today.getFullYear() + String(today.getMonth() + 1).padStart(2, '0') + String(today.getDate()).padStart(2, '0');
      doc.save(`${dateStr}_ELISA_Results.pdf`);
    } catch (e) { alert("PDF Error"); console.error(e); } finally { setIsGeneratingDoc(false); }
  };

  const handleDownloadExcel = async () => {
    if (!fitResult || calculatedData.length === 0) return alert("Please calculate fit first.");
    setIsGeneratingExcel(true);
    try {
      const workbook = new ExcelJS.Workbook();
      workbook.creator = 'ELISA Calculator';
      
      // Sheet 1: Plate Layout
      const ws1 = workbook.addWorksheet('Plate Layout');
      const layoutData = [];
      layoutData.push(['', ...Array.from({ length: 12 }, (_, i) => i + 1)]);
      ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'].forEach((r, i) => {
        const row = [r, ...layout[i].map(c => {
          if (c.type === 'EMPTY') return '';
          if (c.type === 'STD') return `STD${c.id}`;
          if (c.type === 'CTL') return `CTL-${c.id}`;
          return c.id;
        })];
        layoutData.push(row);
      });
      ws1.addRows(layoutData);
      
      // Sheet 2: OD Value Layout
      const ws2 = workbook.addWorksheet('OD Value Layout');
      const odData = [];
      odData.push(['', ...Array.from({ length: 12 }, (_, i) => i + 1)]);
      ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'].forEach((r, i) => {
        const row = [r, ...odValues[i].map(val => val === '' ? '-' : val)];
        odData.push(row);
      });
      ws2.addRows(odData);

      // Sheet 3: Curve Fitting Results
      const ws3 = workbook.addWorksheet('Curve Fitting Results');
      const descA_ex = fitResult.params.a < fitResult.params.d ? "Min Asymptote (Bottom)" : "Max Asymptote (Top)";
      const descD_ex = fitResult.params.a < fitResult.params.d ? "Max Asymptote (Top)" : "Min Asymptote (Bottom)";
      const ec50_ex = fitResult.params.c * Math.pow((Math.pow(2, 1 / fitResult.params.g) - 1), 1 / fitResult.params.b);

      ws3.addRow(['Parameter', 'Value', 'Description']);
      const rSquareRow = ws3.addRow(['R-Squared', fitResult.rSquared, 'Coefficient of Determination']);
      rSquareRow.getCell(2).numFmt = '0.00000';
      const aRow = ws3.addRow(['A', fitResult.params.a, descA_ex]);
      aRow.getCell(2).numFmt = '0.0000';
      const dRow = ws3.addRow(['D', fitResult.params.d, descD_ex]);
      dRow.getCell(2).numFmt = '0.0000';
      const cRow = ws3.addRow(['C', fitResult.params.c, 'Inflection Point']);
      cRow.getCell(2).numFmt = '0.0000';
      const bRow = ws3.addRow(['B', fitResult.params.b, 'Hill Slope']);
      bRow.getCell(2).numFmt = '0.0000';
      const gRow = ws3.addRow(['G', fitResult.params.g, 'Symmetry Factor']);
      gRow.getCell(2).numFmt = '0.0000';
      const ec50Row = ws3.addRow(['EC50', ec50_ex, 'Half Maximal Effective Concentration']);
      ec50Row.getCell(2).numFmt = '0.0000';
      ws3.addRow([]);
      ws3.addRow(['Model Formula', 'y = D + (A - D) / ((1 + (x/C)^B)^G)']);
      ws3.addRow([]);
      
      try {
        await loadScript("https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js");
        const chartContainer = document.getElementById('chart-container');
        if (chartContainer) {
          const canvas = await window.html2canvas(chartContainer, { scale: 2, useCORS: true, logging: false, backgroundColor: '#ffffff' });
          const imgData = canvas.toDataURL('image/png');
          const imageId = workbook.addImage({ base64: imgData, extension: 'png' });
          ws3.addImage(imageId, {
            tl: { col: 0, row: 13 },
            ext: { width: Math.round(canvas.width * 0.4), height: Math.round(canvas.height * 0.4) }
          });
        }
      } catch (e) {
        console.error("Failed to capture chart image for Excel:", e);
      }

      // Sheet 4: Standard Curve Statistics
      const ws4 = workbook.addWorksheet('Standard Curve Statistics');
      ws4.addRow(['ID', 'Well', 'Std Conc.', 'OD', 'Back-calc', 'Avg Conc.', 'SD', 'CV (%)']);
      const stdGroups = {};
      calculatedData.filter(d => d.type === 'STD').forEach(d => { if (!stdGroups[d.id]) stdGroups[d.id] = { wells: [] }; stdGroups[d.id].wells.push(d); });
      Object.keys(stdGroups).sort((a, b) => parseInt(a) - parseInt(b)).forEach(id => {
        const group = stdGroups[id];
        const validConcs = group.wells.map(w => w.conc).filter(c => !isNaN(c));
        const { mean, sd, cv } = calcStats(validConcs);
        const stdDef = stdConcs.find(s => s.id === parseInt(id));
        const theoConc = stdDef ? stdDef.conc : '-';
        group.wells.forEach((well, idx) => {
          ws4.addRow([
            idx === 0 ? `STD ${id}` : '',
            well.well,
            idx === 0 ? theoConc : '',
            Number(well.od),
            isNaN(well.conc) ? "Out of Range" : well.conc,
            idx === 0 ? mean : '',
            idx === 0 ? (sd !== null ? sd : 'NA') : '',
            idx === 0 ? (cv !== null ? cv : 'NA') : ''
          ]);
        });
      });
      ws4.getColumn(4).numFmt = '0.000';
      ws4.getColumn(5).numFmt = '0.000';
      ws4.getColumn(6).numFmt = '0.000';
      ws4.getColumn(7).numFmt = '0.000';
      ws4.getColumn(8).numFmt = '0.0';

      // Sheet 5: Sample Concentrations
      const ws5 = workbook.addWorksheet('Sample Concentrations');
      ws5.addRow(['Well', 'Sample ID', 'OD', 'Concentration', 'Avg Conc.', 'SD', 'CV (%)']);
      const sampleGroups = {};
      calculatedData.filter(d => d.type !== 'STD').forEach(d => {
        const key = `${d.type}-${d.id}`;
        if (!sampleGroups[key]) sampleGroups[key] = { type: d.type, id: d.id, wells: [] };
        sampleGroups[key].wells.push(d);
      });
      const typePriority = { 'CTL': 0, 'UNK': 1, 'BLK': 2 };
      const getPriority = (t) => typePriority[t] ?? 9;
      Object.values(sampleGroups).sort((a, b) => {
        const pA = getPriority(a.type);
        const pB = getPriority(b.type);
        if (pA !== pB) return pA - pB;
        if (a.type === 'CTL' && b.type === 'CTL') {
          if (a.id === 'H' && b.id === 'L') return -1;
          if (a.id === 'L' && b.id === 'H') return 1;
        }
        return String(a.id).localeCompare(String(b.id), undefined, { numeric: true });
      }).forEach(group => {
        const validConcs = group.wells.map(w => w.conc).filter(c => !isNaN(c));
        const { mean, sd, cv } = calcStats(validConcs);
        const displayName = group.type === 'UNK' ? group.id : group.type === 'CTL' ? `CTL-${group.id}` : group.id;
        group.wells.forEach((well, idx) => {
          ws5.addRow([
            well.well,
            displayName,
            Number(well.od),
            isNaN(well.conc) ? "Out of Range" : well.conc,
            idx === 0 ? (isNaN(mean) ? '-' : mean) : '',
            idx === 0 ? (sd === null ? '' : sd) : '',
            idx === 0 ? (cv === null ? '' : cv) : ''
          ]);
        });
      });
      ws5.getColumn(3).numFmt = '0.000';
      ws5.getColumn(4).numFmt = '0.000';
      ws5.getColumn(5).numFmt = '0.000';
      ws5.getColumn(6).numFmt = '0.000';
      ws5.getColumn(7).numFmt = '0.0';

      const buffer = await workbook.xlsx.writeBuffer();
      const today = new Date();
      const dateStr = today.getFullYear() + String(today.getMonth() + 1).padStart(2, '0') + String(today.getDate()).padStart(2, '0');
      saveAs(new Blob([buffer]), `${dateStr}_ELISA_Results.xlsx`);
    } catch (e) {
      alert("Excel Generation Error");
      console.error(e);
    } finally {
      setIsGeneratingExcel(false);
    }
  };

  const renderStdStatsTable = () => {
    if (!calculatedData.length) return null;
    const stdGroups = {};
    calculatedData.filter(d => d.type === 'STD').forEach(d => { if (!stdGroups[d.id]) stdGroups[d.id] = { wells: [] }; stdGroups[d.id].wells.push(d); });
    return (
      <div className="mt-8 bg-white border border-gray-200 rounded-lg overflow-hidden shadow-sm">
        <div className="px-4 py-3 bg-indigo-50 border-b border-indigo-100 font-bold text-indigo-800 text-sm">Standard Curve Statistics</div>
        <div className="overflow-x-auto max-h-[400px] overflow-y-auto">
          <table className="w-full text-sm text-center min-w-[600px] border-collapse border border-slate-200">
            <thead className="bg-gray-50 text-gray-600 border-b sticky top-0 z-10 shadow-sm">
              <tr><th className="px-4 py-2 border border-slate-200 whitespace-nowrap">ID</th><th className="px-4 py-2 border border-slate-200 whitespace-nowrap">Well</th><th className="px-4 py-2 border border-slate-200 whitespace-nowrap">Std Conc.</th><th className="px-4 py-2 border border-slate-200 whitespace-nowrap">OD</th><th className="px-4 py-2 border border-slate-200 whitespace-nowrap">Back-calc</th><th className="px-4 py-2 bg-gray-100 border border-slate-200 whitespace-nowrap">Avg Conc.</th><th className="px-4 py-2 bg-gray-100 border border-slate-200 whitespace-nowrap">SD</th><th className="px-4 py-2 bg-gray-100 border border-slate-200 whitespace-nowrap">CV (%)</th></tr>
            </thead>
            <tbody>
              {Object.keys(stdGroups).sort((a, b) => parseInt(a) - parseInt(b)).map(id => {
                const group = stdGroups[id];
                const validConcs = group.wells.map(w => w.conc).filter(c => !isNaN(c));
                const { mean, sd, cv } = calcStats(validConcs);
                const stdDef = stdConcs.find(s => s.id === parseInt(id));
                const theoConc = stdDef ? stdDef.conc : '-';
                return group.wells.map((well, idx) => (
                  <tr key={well.well} className="hover:bg-gray-50">
                    <td className="px-4 py-2 border border-slate-200 font-bold text-gray-700">{idx === 0 ? `STD ${id}` : ''}</td>
                    <td className="px-4 py-2 border border-slate-200 font-mono text-gray-500">{well.well}</td><td className="px-4 py-2 border border-slate-200 font-bold text-gray-700">{idx === 0 ? theoConc : ''}</td><td className="px-4 py-2 border border-slate-200 font-mono">{well.od.toFixed(3)}</td><td className="px-4 py-2 border border-slate-200 font-mono">{isNaN(well.conc) ? "Out" : well.conc.toFixed(3)}</td>
                    <td className="px-4 py-2 bg-gray-50 border border-slate-200 font-bold text-indigo-700">{idx === 0 ? mean.toFixed(3) : ''}</td><td className="px-4 py-2 bg-gray-50 border border-slate-200">{idx === 0 ? (sd !== null ? sd.toFixed(3) : 'NA') : ''}</td><td className="px-4 py-2 bg-gray-50 border border-slate-200">{idx === 0 ? (cv !== null ? cv.toFixed(1) : 'NA') : ''}</td>
                  </tr>
                ));
              })}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  const renderSampleTable = () => {
    if (!calculatedData.length) return null;
    const sampleGroups = {};
    calculatedData.filter(d => d.type !== 'STD').forEach(d => {
      const key = `${d.type}-${d.id}`;
      if (!sampleGroups[key]) sampleGroups[key] = { type: d.type, id: d.id, wells: [] };
      sampleGroups[key].wells.push(d);
    });

    const typePriority = { 'CTL': 0, 'UNK': 1, 'BLK': 2 };
    const getPriority = (t) => typePriority[t] ?? 9;

    return (
      <div className="mt-8 bg-white border border-gray-200 rounded-lg overflow-hidden shadow-sm">
        <div className="px-4 py-3 bg-amber-50 border-b border-amber-100 font-bold text-amber-800 text-sm">Sample Concentrations</div>
        <div className="max-h-[400px] overflow-y-auto">
          <table className="w-full text-sm text-center min-w-[600px] border-collapse border border-slate-200">
            <thead className="bg-gray-50 border-b sticky top-0 z-10">
              <tr><th className="px-4 py-2 border border-slate-200 whitespace-nowrap">Well</th><th className="px-4 py-2 border border-slate-200 whitespace-nowrap">ID</th><th className="px-4 py-2 border border-slate-200 whitespace-nowrap">OD</th><th className="px-4 py-2 border border-slate-200 whitespace-nowrap">Conc.</th><th className="px-4 py-2 bg-gray-100 border border-slate-200 whitespace-nowrap">Avg Conc.</th><th className="px-4 py-2 bg-gray-100 border border-slate-200 whitespace-nowrap">SD</th><th className="px-4 py-2 bg-gray-100 border border-slate-200 whitespace-nowrap">CV (%)</th></tr>
            </thead>
            <tbody>
              {Object.values(sampleGroups).sort((a, b) => {
                const pA = getPriority(a.type);
                const pB = getPriority(b.type);
                if (pA !== pB) return pA - pB;
                // Sort logic for CTL: H before L
                if (a.type === 'CTL' && b.type === 'CTL') {
                  if (a.id === 'H' && b.id === 'L') return -1;
                  if (a.id === 'L' && b.id === 'H') return 1;
                }
                return String(a.id).localeCompare(String(b.id), undefined, { numeric: true });
              }).map(group => {
                const validConcs = group.wells.map(w => w.conc).filter(c => !isNaN(c));
                const { mean, sd, cv } = calcStats(validConcs);
                const displayName = group.type === 'UNK' ? group.id : group.type === 'CTL' ? `CTL-${group.id}` : group.id;

                return group.wells.map((well, idx) => (
                  <tr key={well.well} className="hover:bg-gray-50">
                    <td className="px-4 py-2 border border-slate-200 font-mono text-gray-500">{well.well}</td>
                    <td className="px-4 py-2 border border-slate-200"><span className={`px-2 py-0.5 rounded text-[10px] font-bold ${group.type === 'CTL' ? 'bg-purple-100 text-purple-700' : group.type === 'UNK' ? 'bg-amber-100 text-amber-700' : 'bg-gray-100'}`}>{displayName}</span></td>
                    <td className="px-4 py-2 border border-slate-200 font-mono text-gray-700">{well.od.toFixed(3)}</td>
                    <td className="px-4 py-2 border border-slate-200 font-mono font-bold text-gray-800">{isNaN(well.conc) ? <span className="text-red-400 text-xs">Error</span> : well.conc.toFixed(3)}</td>
                    <td className="px-4 py-2 bg-gray-50 border border-slate-200 font-bold text-indigo-700">{idx === 0 ? (isNaN(mean) ? '-' : mean.toFixed(3)) : ''}</td>
                    <td className="px-4 py-2 bg-gray-50 border border-slate-200">{idx === 0 ? (sd === null ? '' : sd.toFixed(3)) : ''}</td>
                    <td className={`px-4 py-2 bg-gray-50 border border-slate-200 ${idx === 0 && cv > 15 ? 'text-red-600 font-bold' : ''}`}>{idx === 0 ? (cv === null ? '' : cv.toFixed(1)) : ''}</td>
                  </tr>
                ));
              })}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50 text-slate-800 font-sans">
      <header className="bg-white border-b px-6 py-4 flex items-center justify-between sticky top-0 z-20 shadow-sm">
        <div className="flex items-center gap-2 text-indigo-700"><Activity className="w-6 h-6" /><h1 className="text-xl font-bold tracking-tight">ELISA Calculator</h1></div>
        <div className="flex gap-3"><button onClick={loadDemoData} className="flex items-center gap-2 px-3 py-1.5 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-md"><RefreshCw className="w-4 h-4" /> Load Demo</button><button onClick={handleCalculate} className="flex items-center gap-2 px-4 py-1.5 text-sm bg-indigo-600 hover:bg-indigo-700 text-white rounded-md font-medium shadow"><Calculator className="w-4 h-4" /> Calculate Fit</button></div>
      </header>
      <main className="max-w-7xl mx-auto p-4 md:p-6">
        <div className="mb-6 bg-blue-50 border border-blue-100 rounded-lg p-4 flex items-start gap-3 shadow-sm">
          <Info className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-blue-900 space-y-2">
            <h4 className="font-bold text-blue-800">Disclaimer / 免責聲明</h4>
            <div>
              <p className="mb-1">This tool is provided for reference purposes only. Users are responsible for verifying the accuracy of the data. We assume no liability for any direct or indirect damages resulting from the use of this tool. All calculations are processed locally in your browser; no experimental data is stored on our servers.</p>
              <p>本工具僅供參考使用，使用者應自行核實數據的準確性。本網站不承擔因使用本工具所產生的任何直接或間接責任。所有運算皆於您的瀏覽器端執行，我們不會儲存您的實驗數據。</p>
            </div>
          </div>
        </div>
        <div className="flex border-b border-gray-200 mb-6 overflow-x-auto">{[{ id: 'layout', icon: Edit3, label: '1. Plate Layout' }, { id: 'std', icon: Settings, label: '2. Standard Conc.' }, { id: 'data', icon: FileText, label: '3. Input OD Values' }, { id: 'results', icon: BarChart2, label: '4. Results' }].map(tab => (<button key={tab.id} onClick={() => setActiveTab(tab.id)} className={`px-4 py-3 text-sm font-medium flex items-center gap-2 border-b-2 whitespace-nowrap transition-colors ${activeTab === tab.id ? 'border-indigo-600 text-indigo-600 bg-indigo-50/50' : 'border-transparent text-gray-500 hover:text-gray-700'}`}><tab.icon className="w-4 h-4" /> {tab.label}</button>))}</div>
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 min-h-[600px]">
          {activeTab === 'layout' && (
            <div className="flex flex-col lg:flex-row gap-8"><div className="flex-1"><div className="flex justify-between items-center mb-4"><h3 className="text-lg font-semibold text-gray-800">Plate Layout Configuration</h3><ConfirmButton onConfirm={clearLayout} label="Clear Layout" icon={XCircle} className="text-xs flex items-center gap-1 text-red-500 hover:text-red-700 border border-red-200 bg-red-50 px-2 py-1 rounded" /></div><PlateGrid data={layout} setData={setLayout} type="layout" activeTool={activeTool} /></div><div className="w-full lg:w-72 space-y-4"><div className="bg-indigo-50 p-4 rounded-lg border border-indigo-100"><h4 className="font-bold text-indigo-800 mb-3 text-sm flex items-center gap-2"><Grid size={14} /> Auto Layout Settings</h4><div className="space-y-3 mb-4"><div className="flex justify-between items-center"><span className="text-xs font-semibold text-gray-600">Standard Repeats:</span><select value={repeatSettings.std} onChange={e => setRepeatSettings({ ...repeatSettings, std: parseInt(e.target.value) })} className="text-xs border rounded p-1"><option value="2">2 (e.g. A1, A2)</option><option value="3">3 (e.g. A1, A2, A3)</option></select></div><div className="flex justify-between items-center"><span className="text-xs font-semibold text-gray-600">Control Repeats:</span><select value={repeatSettings.ctl} onChange={e => setRepeatSettings({ ...repeatSettings, ctl: parseInt(e.target.value) })} className="text-xs border rounded p-1"><option value="2">2</option><option value="3">3</option></select></div><div className="flex justify-between items-center"><span className="text-xs font-semibold text-gray-600">Sample Repeats:</span><select value={repeatSettings.sample} onChange={e => setRepeatSettings({ ...repeatSettings, sample: parseInt(e.target.value) })} className="text-xs border rounded p-1"><option value="1">1</option><option value="2">2</option><option value="3">3</option></select></div></div><button onClick={applyAutoLayout} className="w-full py-2 bg-indigo-600 text-white text-xs font-bold rounded shadow hover:bg-indigo-700 flex justify-center items-center gap-2"><Play size={12} fill="currentColor" /> Apply Auto Layout</button><p className="text-[10px] text-indigo-400 mt-2 text-center">Note: This will overwrite current layout.</p></div><div className="bg-amber-50 p-4 rounded-lg border border-amber-200 text-xs text-amber-800"><p className="font-bold mb-1">Tip:</p><p>Click grid cells to edit Well IDs directly (e.g., change S01 to Patient A).</p></div></div></div>
          )}
          {activeTab === 'std' && (
            <div className="max-w-2xl mx-auto"><div className="flex justify-between items-center mb-4"><div className="flex items-center gap-2"><h3 className="text-lg font-semibold flex items-center gap-2">Set Standard Concentrations</h3><span className="text-xs font-normal text-gray-500 bg-gray-100 px-2 py-1 rounded border flex items-center gap-1"><Clipboard size={12} /> Excel Paste Supported</span></div><ConfirmButton onConfirm={clearStandards} label="Clear" icon={XCircle} className="text-xs flex items-center gap-1 text-red-500 hover:text-red-700 border border-red-200 bg-red-50 px-2 py-1 rounded" /></div><div className="border rounded-lg overflow-hidden"><table className="w-full text-sm text-left"><thead className="bg-gray-50 border-b"><tr><th className="px-6 py-3">ID</th><th className="px-6 py-3">Concentration</th><th className="px-6 py-3"></th></tr></thead><tbody className="divide-y divide-gray-100">{stdConcs.map((s, i) => <tr key={i}><td className="px-6 py-2 font-medium text-blue-600">STD {s.id}</td><td className="px-6 py-2"><input type="number" value={s.conc} onChange={(e) => { const n = [...stdConcs]; n[i].conc = parseFloat(e.target.value); setStdConcs(n) }} onPaste={(e) => handleStdPaste(e, i)} className="border rounded px-2 py-1 w-32 focus:ring-2 focus:ring-blue-500 outline-none" /></td><td className="px-6 py-2"><button onClick={() => setStdConcs(stdConcs.filter((_, idx) => idx !== i))} className="text-red-400"><Trash2 className="w-4 h-4" /></button></td></tr>)}</tbody></table><div className="p-3 bg-gray-50 border-t"><button onClick={() => setStdConcs([...stdConcs, { id: stdConcs.length + 1, conc: 0 }])} className="text-indigo-600 text-xs font-bold">+ Add Standard</button></div></div></div>
          )}
          {activeTab === 'data' && (
            <div><div className="flex justify-between items-center mb-4"><h3 className="text-lg font-semibold flex items-center gap-2">Input Raw OD Values <span className="text-xs font-normal text-gray-500 bg-gray-100 px-2 py-1 rounded border flex items-center gap-1"><Clipboard size={12} /> Excel Block Paste Supported</span></h3><ConfirmButton onConfirm={clearODs} label="Clear All Values" icon={XCircle} className="text-xs flex items-center gap-1 text-red-500 hover:text-red-700 border border-red-200 bg-red-50 px-2 py-1 rounded" /></div><PlateGrid data={odValues} setData={setOdValues} type="values" /></div>
          )}
          {activeTab === 'results' && (
            <div className="space-y-6">
              {!fitResult ? (
                <div className="flex flex-col items-center justify-center h-64 text-gray-400 border-2 border-dashed rounded-lg"><Activity className="w-12 h-12 mb-2 opacity-20" /><p>Please enter data and click "Calculate Fit"</p></div>
              ) : (
                <>
                  <div className="flex justify-between items-center">
                    <h2 className="text-lg font-bold text-gray-800">Analysis Results</h2>
                    <div className="flex gap-2">
                      <button onClick={handleDownloadExcel} disabled={isGeneratingExcel} className="flex items-center gap-1 text-xs bg-emerald-600 text-white px-3 py-1.5 rounded hover:bg-emerald-700 shadow-sm disabled:opacity-50">
                        {isGeneratingExcel ? <RefreshCw className="w-3 h-3 animate-spin" /> : <Download className="w-3 h-3" />}
                        {isGeneratingExcel ? "Processing..." : "Download Excel"}
                      </button>
                      <button onClick={handleDownloadPdf} disabled={isGeneratingDoc} className="flex items-center gap-1 text-xs bg-rose-600 text-white px-3 py-1.5 rounded hover:bg-rose-700 shadow-sm disabled:opacity-50">
                        {isGeneratingDoc ? <RefreshCw className="w-3 h-3 animate-spin" /> : <Download className="w-3 h-3" />}
                        {isGeneratingDoc ? "Processing..." : "Download PDF"}
                      </button>
                    </div>
                  </div>
                  <div className="flex flex-col lg:flex-row gap-6 mb-6">
                    <div className="lg:w-1/3"><div className="bg-white border border-slate-200 rounded-lg p-5 shadow-sm h-full"><h4 className="font-bold text-slate-700 mb-4 text-base border-b pb-2">Fit Parameters</h4><div className="mb-6 p-4 bg-green-50 border border-green-100 rounded-lg text-center"><div className="text-xs text-green-600 font-semibold uppercase tracking-wider mb-1">Goodness of Fit (R²)</div><div className="text-3xl font-bold text-green-700 tracking-tight">{fitResult.rSquared.toFixed(5)}</div></div><div className="mb-4 p-3 bg-slate-50 border border-slate-200 rounded text-center"><div className="text-xs text-slate-500 mb-1">5PL Model Formula</div><div className="font-mono text-xs font-bold text-indigo-700 mt-2">y = D + (A - D) / ((1 + (x/C)^B)^G)</div></div><div className="space-y-3">{['a', 'd', 'c', 'b', 'g'].map(p => <div key={p} className="flex justify-between items-center text-sm"><span className="text-slate-500">{p.toUpperCase()}</span><span className="font-mono font-bold text-slate-700 bg-slate-50 px-2 py-0.5 rounded">{fitResult.params[p].toFixed(4)}</span></div>)}<div className="flex justify-between items-center text-sm"><span className="text-slate-500 font-bold text-indigo-700">EC50</span><span className="font-mono font-bold text-indigo-700 bg-indigo-50 px-2 py-0.5 rounded border border-indigo-200">{(fitResult.params.c * Math.pow((Math.pow(2, 1/fitResult.params.g) - 1), 1/fitResult.params.b)).toFixed(4)}</span></div></div></div></div>
                    <div className="lg:w-2/3"><div className="bg-white border border-slate-200 rounded-lg p-4 shadow-sm h-full flex flex-col"><div className="flex justify-between items-center mb-2"><h4 className="font-bold text-slate-700">Standard Curve</h4><div className="flex bg-gray-100 rounded p-1"><button onClick={() => setChartScale('log')} className={`px-2 py-1 text-xs rounded ${chartScale === 'log' ? 'bg-white shadow text-indigo-600 font-bold' : 'text-slate-500'}`}>Log</button><button onClick={() => setChartScale('linear')} className={`px-2 py-1 text-xs rounded ${chartScale === 'linear' ? 'bg-white shadow text-indigo-600 font-bold' : 'text-slate-500'}`}>Linear</button></div></div><div id="chart-container" className="flex-1 min-h-[400px] h-[400px] w-full"><ResponsiveContainer width="100%" height="100%"><ComposedChart margin={{ top: 20, right: 30, bottom: 50, left: 20 }}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="x" type="number" scale={chartScale} domain={[0.1, 1000]} allowDataOverflow ticks={chartScale === 'log' ? [0.1, 1, 10, 100, 1000] : null} tickFormatter={t => chartScale === 'log' ? Number(t).toString() : parseInt(t)} label={{ value: `Concentration`, position: 'insideBottom', offset: -10 }} /><YAxis dataKey="y" type="number" label={{ value: 'OD', angle: -90, position: 'insideLeft' }} /><Tooltip labelFormatter={l => `Conc: ${Number(l).toFixed(3)}`} formatter={v => v.toFixed(3)} /><Legend verticalAlign="top" wrapperStyle={{ paddingBottom: '20px' }} /><Line data={getChartData.curve} type="monotone" dataKey="y" stroke="#4f46e5" dot={false} name="Fit Curve" isAnimationActive={false} /><Scatter data={getChartData.scatter} fill="#ef4444" name="Standards" /></ComposedChart></ResponsiveContainer></div></div></div>
                  </div>
                  {renderStdStatsTable()}
                  {renderSampleTable()}
                </>
              )}
            </div>
          )}
        </div>
      </main>
      <footer className="py-6 text-center text-sm text-gray-400">
        Designed by Bio Preventive Medicine
      </footer>
    </div>
  );
}
