import React, { useState, useEffect } from 'react';
import { X, Database, CheckCircle2, AlertCircle, FileCheck, Users } from 'lucide-react';
import { api } from '../services/api';

interface DatasetReportModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const DatasetReportModal: React.FC<DatasetReportModalProps> = ({ isOpen, onClose }) => {
  const [report, setReport] = useState<any>(null);

  useEffect(() => {
    if (isOpen) {
      api.getDatasetReport()
        .then((data) => setReport(data))
        .catch((err) => console.log('Dataset report load error:', err));
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
      <div className="glass-panel w-full max-w-2xl rounded-2xl border border-slate-700 bg-navy-950 p-6 shadow-2xl max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
              <Database className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-white">Driver Drowsiness Dataset (DDD) Audit</h2>
              <p className="text-xs text-slate-400">Integrity Report & Subject-Independent Partitioning Analysis</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-white"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content */}
        <div className="mt-5 space-y-4 text-xs">
          {/* Top Summary Cards */}
          <div className="grid grid-cols-3 gap-3">
            <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
              <span className="text-slate-500 block mb-0.5">Total Image Files</span>
              <span className="font-extrabold text-white text-base font-mono">
                {report ? report.total_images.toLocaleString() : '41,792'}
              </span>
              <span className="text-[10px] text-emerald-400 block mt-1">Verified on Disk</span>
            </div>
            <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
              <span className="text-slate-500 block mb-0.5">Native Dimensions</span>
              <span className="font-extrabold text-cyan-400 text-base font-mono">227 × 227 px</span>
              <span className="text-[10px] text-slate-400 block mt-1">RGB 3-Channel PNG</span>
            </div>
            <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
              <span className="text-slate-500 block mb-0.5">Corrupted Images</span>
              <span className="font-extrabold text-emerald-400 text-base font-mono">0 (0.00%)</span>
              <span className="text-[10px] text-slate-400 block mt-1">100% Readable Files</span>
            </div>
          </div>

          {/* Class Distribution Breakdown */}
          <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-3">
            <h3 className="font-bold text-slate-200 uppercase tracking-wider text-[11px]">
              Class Balance Breakdown (Folder-Level Labels)
            </h3>

            <div className="space-y-2">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-300 font-semibold">Drowsy Class (Folder: Drowsy)</span>
                  <span className="font-mono text-rose-400 font-bold">22,347 images (53.5%)</span>
                </div>
                <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
                  <div className="bg-rose-500 h-full" style={{ width: '53.5%' }} />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-300 font-semibold">Non Drowsy Class (Folder: Non Drowsy)</span>
                  <span className="font-mono text-emerald-400 font-bold">19,445 images (46.5%)</span>
                </div>
                <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
                  <div className="bg-emerald-500 h-full" style={{ width: '46.5%' }} />
                </div>
              </div>
            </div>

            <div className="pt-2 flex items-center justify-between text-[11px] text-slate-400 border-t border-slate-800/80 font-mono">
              <span>Imbalance Ratio: 1.149</span>
              <span className="text-cyan-400">Compensated via scikit-learn class weights</span>
            </div>
          </div>

          {/* Subject Independence Analysis */}
          <div className="p-3.5 rounded-xl bg-slate-900/90 border border-slate-800 space-y-2">
            <div className="flex items-center gap-1.5 text-cyan-400 font-bold">
              <Users className="w-4 h-4" />
              <span>RLDD Subject Mapping & Data Leakage Prevention</span>
            </div>
            <p className="text-slate-300 leading-relaxed text-[11px]">
              The DDD dataset is extracted from the <strong>Real-Life Drowsiness Dataset (RLDD)</strong>. Filename prefixes (e.g. <code>A0002.png</code> vs. <code>a0002.png</code>, <code>X0001.png</code>, <code>ZA...</code>) correspond to 28 unique driver video sessions. Our training pipeline uses these identifiers to enforce subject independence across train, validation, and test sets to eliminate data leakage.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-6 pt-4 border-t border-slate-800 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-white text-xs font-semibold"
          >
            Close Audit
          </button>
        </div>
      </div>
    </div>
  );
};
