import { useState, useRef, useCallback, useEffect, memo, useMemo } from "react";
import { useAuth, useDashboard, usePatients, useScan, useAnalysis, useReport, useSettings, useBlobImage } from "../services/hooks";
import { AUTH_EXPIRED_EVENT, reportsApi, patientsApi } from "../services/api";
import { toast, Toaster } from "sonner";
import {
  AreaChart, Area, BarChart, Bar, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend
} from "recharts";
import {
  Eye, LayoutDashboard, ScanEye, FileText, Users, Settings,
  Bell, Search, ChevronDown, Upload, Activity,
  CheckCircle, AlertTriangle, XCircle, Download, Filter, MoreHorizontal,
  Brain, Microscope, HeartPulse, ChevronRight, Menu, X,
  LogOut, User, Lock, Moon, Sun, Mail, Phone, Building, Plus,
  RefreshCw, Calendar, Layers, Target, MessageSquare, Printer, Trash2
} from "lucide-react";

// ─── Icon Map (API returns icon names as strings) ────────────────────────────
const ICON_MAP: Record<string, any> = {
  Calendar, HeartPulse, Microscope, CheckCircle, AlertTriangle, XCircle,
  Brain, Activity, Eye, FileText, Users, Settings, Bell, Search, Upload,
  Download, Filter, RefreshCw, Layers, Target, MessageSquare, Printer,
  Trash2, Mail, Phone, Building, Plus, ScanEye, User, Lock,
  Moon, Sun, LogOut, ChevronRight, Menu, X,
};

// ─── Types ───────────────────────────────────────────────────────────────────
type Page = "login" | "signup" | "dashboard" | "scan" | "analysis" | "reports" | "patients" | "settings";

interface AuthUser {
  name: string;
  role: string;
  initials: string;
  email: string;
}

// ─── Data ────────────────────────────────────────────────────────────────────
interface Patient {
  id: string; name: string; age: number; condition: string; severity: string;
  lastScan: string; status: string; risk: string; contact: string; email: string; mobile: string;
  date_of_birth: string; gender: string; notes: string;
}

interface PatientDetails {
  fullName: string; dob: string; eye: string; gender: string; mobile: string; notes: string; patient_id: string;
}

const EMPTY_PATIENT_DETAILS: PatientDetails = { fullName: "", dob: "", eye: "Left Eye (OS)", gender: "", mobile: "", notes: "", patient_id: "" };

// ─── Helpers ──────────────────────────────────────────────────────────────────
function cn(...classes: (string | undefined | false | null)[]) {
  return classes.filter(Boolean).join(" ");
}

function toTitleCase(str: string | undefined | null): string {
  if (!str) return "";
  return str.replace(/\w\S*/g, (txt) => txt.charAt(0).toUpperCase() + txt.slice(1).toLowerCase());
}

function riskColor(r: string | undefined): "error" | "warning" | "success" | "info" {
  if (!r) return "info";
  if (r === "High" || r === "Critical") return "error";
  if (r === "Medium" || r === "Moderate") return "warning";
  if (r === "Low" || r === "Stable") return "success";
  const num = parseFloat(r);
  if (!isNaN(num)) {
    return num >= 7.0 ? "error" : num >= 4.0 ? "warning" : "success";
  }
  return "info";
}

function applyTheme(dark: boolean) {
  document.documentElement.classList.toggle("dark", dark);
}

const Skeleton = memo(function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse bg-muted rounded ${className}`} />;
});

const SkeletonCard = memo(function SkeletonCard({ lines = 3 }: { lines?: number }) {
  return (
    <div className="bg-card rounded-2xl border border-border p-6 space-y-3">
      <Skeleton className="h-4 w-1/3" />
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton key={i} className="h-3 w-full" />
      ))}
    </div>
  );
});

const Badge = memo(function Badge({ children, variant = "default" }: { children: React.ReactNode; variant?: "default" | "success" | "warning" | "error" | "info" }) {
  const styles = {
    default: "bg-blue-50 text-blue-700 border-blue-100",
    success: "bg-emerald-50 text-emerald-700 border-emerald-100",
    warning: "bg-amber-50 text-amber-700 border-amber-100",
    error: "bg-red-50 text-red-700 border-red-100",
    info: "bg-indigo-50 text-indigo-700 border-indigo-100",
  };
  return (
    <span className={cn("inline-flex items-center px-2 py-0.5 rounded-md text-xs font-medium border", styles[variant])}>
      {children}
    </span>
  );
});

const KpiCard = memo(function KpiCard({ title, value, change, icon: Icon, color, onClick }: { title: string; value: string; change: string; icon: any; color: string; onClick?: () => void }) {
  const positive = change.startsWith("+");
  return (
    <div
      onClick={onClick}
      className={cn(
        "bg-card rounded-2xl border border-border p-5 shadow-sm",
        onClick && "cursor-pointer"
      )}
    >
      <div className="flex items-start justify-between mb-4">
        <div className={cn("w-10 h-10 rounded-xl flex items-center justify-center", color)}>
          <Icon size={18} className="text-white" />
        </div>
        <span className={cn("text-xs font-semibold px-2 py-1 rounded-full", positive ? "bg-emerald-50 text-emerald-600" : "bg-red-50 text-red-600")}>
          {change}
        </span>
      </div>
      <p className="text-2xl font-bold text-foreground">{value}</p>
      <p className="text-sm text-muted-foreground mt-1">{title}</p>
    </div>
  );
});

// ─── Modal ────────────────────────────────────────────────────────────────────
function Modal({ open, onClose, title, children, width = "max-w-lg" }: {
  open: boolean; onClose: () => void; title: string; children: React.ReactNode; width?: string;
}) {
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/40 backdrop-blur-[2px]" onClick={onClose} />
      <div className={cn("relative bg-card rounded-2xl shadow-2xl w-full border border-border flex flex-col max-h-[90vh]", width)}>
        <div className="flex items-center justify-between px-6 py-4 border-b border-border">
          <h2 className="font-bold text-foreground text-base">{title}</h2>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground transition-colors">
            <X size={18} />
          </button>
        </div>
        <div className="overflow-y-auto flex-1">{children}</div>
      </div>
    </div>
  );
}

// ─── Drawer (right slide-in) ──────────────────────────────────────────────────
function Drawer({ open, onClose, title, children }: {
  open: boolean; onClose: () => void; title: string; children: React.ReactNode;
}) {
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [open, onClose]);

  return (
    <div className={cn("fixed inset-0 z-[60] flex justify-end", !open && "pointer-events-none")}>
      <div
        className={cn("absolute inset-0 bg-black/40 backdrop-blur-[2px] transition-opacity duration-300", open ? "opacity-100" : "opacity-0")}
        onClick={onClose}
      />
      <div className={cn(
        "relative bg-card w-full max-w-md shadow-2xl flex flex-col transition-transform duration-300 ease-in-out",
        open ? "translate-x-0" : "translate-x-full"
      )}>
        <div className="flex items-center justify-between px-6 py-4 border-b border-border flex-shrink-0">
          <h2 className="font-bold text-foreground text-base">{title}</h2>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground transition-colors">
            <X size={18} />
          </button>
        </div>
        <div className="overflow-y-auto flex-1">{children}</div>
      </div>
    </div>
  );
}

// ─── Add Patient Modal ────────────────────────────────────────────────────────
function AddPatientModal({ open, onClose, onAdd }: {
  open: boolean; onClose: () => void; onAdd: (data: { name: string; age: number; contact?: string; email?: string; condition?: string; status?: string }) => Promise<any>;
}) {
  const [form, setForm] = useState({ name: "", age: "", contact: "", email: "", condition: "", status: "Active" });
  const [submitting, setSubmitting] = useState(false);
  const set = (k: string, v: string) => setForm(f => ({ ...f, [k]: v }));

  const nameValid = /^[A-Za-z.\s\-']+$/.test(form.name.trim());

  const handleSubmit = async () => {
    if (!form.name.trim() || !form.age) {
      toast.error("Name and age are required.");
      return;
    }
    if (!nameValid) {
      toast.error("Name must contain only letters, dots, hyphens, and spaces.");
      return;
    }
    const ageNum = parseInt(form.age);
    if (isNaN(ageNum) || ageNum < 0 || ageNum > 120) {
      toast.error("Age must be between 0 and 120 years.");
      return;
    }
    setSubmitting(true);
    try {
      const formattedName = toTitleCase(form.name.trim());
      await onAdd({
        name: formattedName, age: ageNum,
        contact: form.contact || undefined, email: form.email || undefined,
        condition: form.condition, status: form.status,
      });
      setForm({ name: "", age: "", contact: "", email: "", condition: "", status: "Active" });
      onClose();
      toast.success(`Patient ${formattedName} added successfully.`);
    } catch (e: any) {
      toast.error(e.message || "Failed to add patient");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Modal open={open} onClose={onClose} title="Add New Patient">
      <div className="px-6 py-5 space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-xs font-semibold text-foreground block mb-1.5">Full Name <span className="text-red-500">*</span></label>
            <input value={form.name} onChange={e => set("name", e.target.value)} placeholder="Patient full name"
              className="w-full px-3 py-2 text-sm bg-muted border border-border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500" />
          </div>
          <div>
            <label className="text-xs font-semibold text-foreground block mb-1.5">Age <span className="text-red-500">*</span></label>
            <input value={form.age} onChange={e => set("age", e.target.value)} type="number" min={1} max={120} placeholder="e.g. 65"
              className="w-full px-3 py-2 text-sm bg-muted border border-border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500" />
          </div>
        </div>
        <div>
          <label className="text-xs font-semibold text-foreground block mb-1.5">Contact Number</label>
          <input value={form.contact} onChange={e => set("contact", e.target.value)} placeholder="+1 (555) 000-0000"
            className="w-full px-3 py-2 text-sm bg-muted border border-border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500" />
        </div>
        <div>
          <label className="text-xs font-semibold text-foreground block mb-1.5">Email</label>
          <input value={form.email} onChange={e => set("email", e.target.value)} type="email" placeholder="patient@email.com"
            className="w-full px-3 py-2 text-sm bg-muted border border-border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500" />
        </div>
        <div>
          <label className="text-xs font-semibold text-foreground block mb-1.5">Diagnosis</label>
          <select value={form.condition} onChange={e => set("condition", e.target.value)}
            className="w-full px-3 py-2 text-sm bg-muted border border-border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500">
            {["", "Diabetic Retinopathy", "Glaucoma", "AMD", "Hypertensive Retinopathy", "Healthy"].map(c => (
              <option key={c} value={c}>{c || "Select diagnosis"}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="text-xs font-semibold text-foreground block mb-1.5">Status</label>
          <div className="flex gap-2">
            {["Active", "Stable", "Critical"].map(s => (
              <button key={s} onClick={() => set("status", s)}
                className={cn("flex-1 py-2 rounded-xl text-xs font-semibold border transition-all",
                  form.status === s ? "bg-blue-50 border-blue-400 text-blue-700" : "border-border text-muted-foreground hover:bg-muted"
                )}>
                {s}
              </button>
            ))}
          </div>
        </div>
      </div>
      <div className="px-6 pb-5 flex gap-2 justify-end border-t border-border pt-4">
        <button onClick={onClose} className="border border-border text-sm font-medium px-4 py-2 rounded-xl hover:bg-muted transition-colors">Cancel</button>
        <button onClick={handleSubmit} className="bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold px-5 py-2 rounded-xl transition-colors flex items-center gap-1.5">
          <Plus size={14} /> Add Patient
        </button>
      </div>
    </Modal>
  );
}

// ─── Patient Detail Drawer ────────────────────────────────────────────────────
function PatientDetailDrawer({ patient, open, onClose, onNav, onViewReport, onNewScan }: {
  patient: Patient | null; open: boolean; onClose: () => void; onNav: (p: Page) => void;
  onViewReport: (scanId: string) => void; onNewScan: (patient: Patient) => void;
}) {
  const [downloading, setDownloading] = useState(false);
  const [history, setHistory] = useState<any[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  useEffect(() => {
    if (!patient) return;
    setHistory([]);
    setHistoryLoading(true);
    patientsApi.get(patient.id)
      .then((d) => setHistory(d.scan_history ?? []))
      .catch(() => {})
      .finally(() => setHistoryLoading(false));
  }, [patient?.id, open]);

  if (!patient) return null;

  const handleDownloadReport = async () => {
    setDownloading(true);
    toast.loading("Generating PDF report…", { id: "pdf" });
    try {
      const blob = await patientsApi.downloadReportPdf(patient.id);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${patient.name.replace(/\s+/g, "_")}_report.pdf`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      toast.success(`Report downloaded for ${patient.name}.`, { id: "pdf" });
    } catch (e: any) {
      toast.error(e.message || "PDF download failed.", { id: "pdf" });
    } finally {
      setDownloading(false);
    }
  };

  return (
    <Drawer open={open} onClose={onClose} title="Patient Profile">
      <div className="p-6 space-y-5">
        {/* Avatar + name */}
        <div className="flex items-center gap-4">
          <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center flex-shrink-0">
            <span className="text-white text-lg font-bold">{patient.name.split(" ").map(n => n[0]).join("")}</span>
          </div>
          <div>
            <p className="font-bold text-foreground text-base">{toTitleCase(patient.name)}</p>
            <p className="text-xs text-muted-foreground font-mono">{patient.id}</p>
            <div className="mt-1 flex items-center gap-1.5">
              <Badge variant={(patient.status === "Critical" ? "error" : patient.status === "Active" ? "info" : "success") as any}>{patient.status}</Badge>
              <Badge variant={riskColor(patient.risk) as any}>{patient.risk} Risk</Badge>
            </div>
          </div>
        </div>

        {/* Info grid */}
        <div className="bg-muted/40 rounded-xl p-4 text-xs">
          <div className="grid grid-cols-2 gap-3">
            {[
              ["Age", patient.age > 0 && patient.age <= 120 ? `${patient.age} years` : "—"],
              ["Diagnosis", patient.condition],
              ["Severity", patient.severity],
              ["Last Scan", patient.lastScan],
              ["Contact", patient.mobile || "—"],
            ].map(([k, v]) => (
              <div key={k}>
                <p className="text-muted-foreground mb-0.5">{k}</p>
                <p className="font-semibold text-foreground truncate">{v}</p>
              </div>
            ))}
          </div>
          {patient.notes && (
            <div className="mt-3 pt-3 border-t border-border/50">
              <p className="text-muted-foreground mb-0.5">Clinical Notes</p>
              <p className="font-semibold text-foreground">{patient.notes}</p>
            </div>
          )}
        </div>

        {/* Scan history */}
        <div>
          <h4 className="font-semibold text-foreground text-sm mb-3">Scan History</h4>
          {historyLoading ? (
            <p className="text-sm text-muted-foreground">Loading scan history…</p>
          ) : history.length === 0 ? (
            <p className="text-sm text-muted-foreground">No scans recorded for this patient yet.</p>
          ) : (
            <div className="space-y-2">
              {history.map((s, i) => (
                <div key={i} className="flex items-center justify-between bg-card border border-border rounded-xl px-4 py-3">
                  <div>
                    <p className="text-xs font-semibold text-foreground">{s.result}</p>
                    <p className="text-xs text-muted-foreground">{s.date} · {s.type}</p>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="text-right">
                      <p className="text-xs font-bold text-blue-600">{s.confidence}</p>
                      <Badge variant={s.status === "Normal" ? "success" : "warning"}>{s.status}</Badge>
                    </div>
                    {s.scan_id && (
                      <button onClick={() => { onViewReport(s.scan_id); onClose(); }}
                        title="View report"
                        className="p-1.5 rounded-lg hover:bg-muted transition-colors text-muted-foreground hover:text-foreground cursor-pointer">
                        <FileText size={14} />
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="grid grid-cols-2 gap-2 pt-2">
          <button onClick={() => { onNewScan(patient); onClose(); }}
            className="flex items-center justify-center gap-1.5 border border-border text-sm font-medium px-3 py-2.5 rounded-xl hover:bg-muted transition-colors">
            <ScanEye size={14} /> New Scan
          </button>
          <button onClick={handleDownloadReport} disabled={downloading}
            className="flex items-center justify-center gap-1.5 bg-blue-600 text-white text-sm font-semibold px-3 py-2.5 rounded-xl hover:bg-blue-700 transition-colors disabled:opacity-60 cursor-pointer">
            {downloading ? <><RefreshCw size={14} className="animate-spin" /> Generating…</> : <><Download size={14} /> Download Report</>}
          </button>
        </div>
      </div>
    </Drawer>
  );
}

// ─── Filter Popover (Patients) ────────────────────────────────────────────────
function FilterPopover({ open, onClose, filters, onChange }: {
  open: boolean; onClose: () => void;
  filters: { diagnosis: string; risk: string; dateFrom: string; dateTo: string };
  onChange: (f: { diagnosis: string; risk: string; dateFrom: string; dateTo: string }) => void;
}) {
  const [local, setLocal] = useState(filters);
  const set = (k: string, v: string) => setLocal(f => ({ ...f, [k]: v }));

  if (!open) return null;
  return (
    <div className="absolute right-0 top-11 z-50 w-72 bg-card rounded-2xl border border-border shadow-2xl p-4 space-y-4">
      <div className="fixed inset-0 -z-10" onClick={onClose} />
      <p className="text-xs font-bold text-foreground uppercase tracking-wide">Advanced Filters</p>
      <div>
        <label className="text-xs font-semibold text-foreground block mb-1.5">Diagnosis</label>
        <select value={local.diagnosis} onChange={e => set("diagnosis", e.target.value)}
          className="w-full px-3 py-2 text-sm bg-muted border border-border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500">
          <option value="">All Diagnoses</option>
          {["Diabetic Retinopathy", "Glaucoma", "AMD", "Hypertensive Retinopathy", "Healthy"].map(c => <option key={c}>{c}</option>)}
        </select>
      </div>
      <div>
        <label className="text-xs font-semibold text-foreground block mb-1.5">Risk Level</label>
        <div className="flex gap-1.5">
          {["", "High", "Medium", "Low"].map(r => (
            <button key={r} onClick={() => set("risk", r)}
              className={cn("flex-1 py-1.5 rounded-lg text-xs font-semibold border transition-all",
                local.risk === r ? "bg-blue-50 border-blue-400 text-blue-700" : "border-border text-muted-foreground hover:bg-muted"
              )}>
              {r || "All"}
            </button>
          ))}
        </div>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="text-xs font-semibold text-foreground block mb-1.5">Date From</label>
          <input type="date" value={local.dateFrom} onChange={e => set("dateFrom", e.target.value)}
            className="w-full px-3 py-2 text-xs bg-muted border border-border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500" />
        </div>
        <div>
          <label className="text-xs font-semibold text-foreground block mb-1.5">Date To</label>
          <input type="date" value={local.dateTo} onChange={e => set("dateTo", e.target.value)}
            className="w-full px-3 py-2 text-xs bg-muted border border-border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500" />
        </div>
      </div>
      <div className="flex gap-2 pt-1">
        <button onClick={() => { setLocal({ diagnosis: "", risk: "", dateFrom: "", dateTo: "" }); onChange({ diagnosis: "", risk: "", dateFrom: "", dateTo: "" }); }}
          className="flex-1 border border-border text-xs font-medium py-2 rounded-xl hover:bg-muted transition-colors">
          Clear
        </button>
        <button onClick={() => { onChange(local); onClose(); toast.success("Filters applied"); }}
          className="flex-1 bg-blue-600 text-white text-xs font-semibold py-2 rounded-xl hover:bg-blue-700 transition-colors">
          Apply
        </button>
      </div>
    </div>
  );
}

// ─── Sidebar ──────────────────────────────────────────────────────────────────
const navItems = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "scan", label: "Retina Scan", icon: ScanEye },
  { id: "analysis", label: "AI Analysis", icon: Brain },
  { id: "reports", label: "Reports", icon: FileText },
  { id: "patients", label: "Patients", icon: Users },
  { id: "settings", label: "Settings", icon: Settings },
];

function Sidebar({ current, onNav, collapsed, onToggle, mobileOpen, onMobileClose }: {
  current: Page; onNav: (p: Page) => void; collapsed: boolean; onToggle: () => void;
  mobileOpen?: boolean; onMobileClose?: () => void;
}) {
  const handleNav = (p: Page) => { onNav(p); onMobileClose?.(); };
  return (
    <>
      {/* Mobile overlay backdrop */}
      {mobileOpen && (
        <div className="fixed inset-0 bg-black/40 z-40 md:hidden" onClick={onMobileClose} />
      )}
    <aside className={cn(
      "no-print h-screen flex flex-col bg-card border-r border-border transition-all duration-300 flex-shrink-0",
      collapsed ? "md:w-16" : "md:w-60",
      // Mobile: fixed overlay drawer
      "fixed md:static top-0 left-0 z-50 md:z-auto",
      mobileOpen ? "flex w-72" : "hidden md:flex"
    )}>
      <div className="flex items-center justify-between h-16 px-4 border-b border-border">
        {!collapsed && (
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center">
              <Eye size={14} className="text-white" />
            </div>
            <span className="font-bold text-sm text-foreground">RetinaSense</span>
            <span className="text-xs font-semibold text-blue-600 bg-blue-50 px-1.5 py-0.5 rounded">AI</span>
          </div>
        )}
        {collapsed && (
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center mx-auto">
            <Eye size={14} className="text-white" />
          </div>
        )}
      </div>

      <nav className="flex-1 py-4 px-2 space-y-0.5 overflow-y-auto">
        {navItems.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => handleNav(id as Page)}
            className={cn(
              "w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all",
              current === id
                ? "bg-blue-50 text-blue-700"
                : "text-muted-foreground hover:bg-muted hover:text-foreground"
            )}
          >
            <Icon size={17} className="flex-shrink-0" />
            {!collapsed && <span>{label}</span>}
            {!collapsed && current === id && <span className="ml-auto w-1.5 h-1.5 rounded-full bg-blue-600" />}
          </button>
        ))}
      </nav>

      <div className="px-2 pb-4 border-t border-border pt-3">
        <button
          onClick={onToggle}
          className={cn(
            "w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium text-muted-foreground hover:bg-muted hover:text-foreground transition-all",
            collapsed && "justify-center"
          )}
        >
          {collapsed ? <Menu size={17} /> : <><Menu size={17} /><span>Collapse Sidebar</span></>}
        </button>
      </div>
    </aside>
    </>
  );
}

// ─── Navbar ───────────────────────────────────────────────────────────────────
const PAGE_LABELS: Record<Page, string> = {
  dashboard: "Dashboard",
  scan:      "Retina Scan",
  analysis:  "AI Analysis",
  reports:   "Reports",
  patients: "Patients",
  settings: "Settings",
  login:     "Login",
};

function Navbar({
  page, onNav, isLoggedIn, user, onLogin, onLogout, onMobileMenuOpen,
}: {
  page: Page;
  onNav: (p: Page) => void;
  isLoggedIn: boolean;
  user: AuthUser;
  onLogin: () => void;
  onLogout: () => void;
  onMobileMenuOpen?: () => void;
}) {
  const [dropdownOpen, setDropdownOpen] = useState(false);

  return (
    <header className="no-print h-14 bg-card border-b border-border flex items-center justify-between px-4 md:px-5 flex-shrink-0 z-30">

      {/* Left — hamburger (mobile) + page label */}
      <div className="flex items-center gap-3">
        <button
          className="md:hidden text-muted-foreground hover:text-foreground transition-colors"
          onClick={onMobileMenuOpen}
        >
          <Menu size={20} />
        </button>
        <span className="text-xl font-bold text-foreground tracking-tight">
          {PAGE_LABELS[page] ?? ""}
        </span>
      </div>

      {/* Right — Auth only (minimal) */}
      <div className="flex items-center">
        {!isLoggedIn ? (
          <button
            onClick={onLogin}
            className="flex items-center gap-1.5 bg-blue-600 hover:bg-blue-700 active:scale-95 text-white text-xs font-semibold px-3.5 py-2 rounded-lg transition-colors duration-200"
          >
            <Lock size={12} /> Login
          </button>
        ) : (
          <div className="relative">
            <button
              onClick={() => setDropdownOpen(o => !o)}
              className="flex items-center gap-2 pl-1 pr-2 py-1 rounded-xl hover:bg-muted transition-colors"
            >
              <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center">
                <span className="text-white text-xs font-bold">{user.initials}</span>
              </div>
              <span className="text-xs font-semibold text-foreground hidden sm:block">{user.name}</span>
              <ChevronDown size={12} className={cn("text-muted-foreground transition-transform hidden sm:block", dropdownOpen && "rotate-180")} />
            </button>

            {dropdownOpen && (
              <>
                <div className="fixed inset-0" onClick={() => setDropdownOpen(false)} />
                <div className="absolute right-0 top-11 w-52 bg-card rounded-2xl border border-border shadow-xl z-50 overflow-hidden py-1">
                  <div className="px-4 py-3 border-b border-border">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center flex-shrink-0">
                        <span className="text-white text-xs font-bold">{user.initials}</span>
                      </div>
                      <div className="min-w-0">
                        <p className="text-xs font-semibold text-foreground truncate">{user.name}</p>
                        <p className="text-xs text-muted-foreground truncate">{user.email}</p>
                      </div>
                    </div>
                  </div>
                  <div className="py-1">
                    {[
                      { icon: User,     label: "View Profile" },
                      { icon: Settings, label: "Settings"     },
                    ].map(({ icon: Icon, label }) => (
                      <button
                        key={label}
                        onClick={() => { onNav("settings"); setDropdownOpen(false); }}
                        className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-foreground hover:bg-muted transition-colors"
                      >
                        <Icon size={14} className="text-muted-foreground" />
                        {label}
                      </button>
                    ))}
                  </div>
                  <div className="border-t border-border py-1">
                    <button
                      onClick={() => { onLogout(); setDropdownOpen(false); }}
                      className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-red-600 hover:bg-red-50 transition-colors"
                    >
                      <LogOut size={14} /> Logout
                    </button>
                  </div>
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </header>
  );
}

// ─── Dashboard Shell ──────────────────────────────────────────────────────────
function AppShell({
  page, onNav, isLoggedIn, user, onLogin, onLogout, children,
}: {
  page: Page;
  onNav: (p: Page) => void;
  isLoggedIn: boolean;
  user: AuthUser;
  onLogin: () => void;
  onLogout: () => void;
  children: React.ReactNode;
}) {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <Sidebar
        current={page} onNav={onNav}
        collapsed={collapsed} onToggle={() => setCollapsed(!collapsed)}
        mobileOpen={mobileOpen} onMobileClose={() => setMobileOpen(false)}
      />
      <div className="flex-1 flex flex-col overflow-hidden min-w-0">
        <Navbar
          page={page} onNav={onNav}
          isLoggedIn={isLoggedIn} user={user}
          onLogin={onLogin} onLogout={onLogout}
          onMobileMenuOpen={() => setMobileOpen(true)}
        />
        <main className="flex-1 overflow-y-auto p-3 md:p-6">{children}</main>
      </div>
    </div>
  );
}

// ─── Page: Landing ────────────────────────────────────────────────────────────
const Spinner = memo(function Spinner({ size = 16 }: { size?: number }) {
  return (
    <svg className="animate-spin" width={size} height={size} viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
    </svg>
  );
});

function ForgotPasswordModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [fpEmail, setFpEmail] = useState("");
  const [sent, setSent] = useState(false);
  const [sending, setSending] = useState(false);

  const handleSend = () => {
    if (!fpEmail.trim()) return;
    setSending(true);
    setTimeout(() => { setSending(false); setSent(true); }, 900);
  };

  const handleClose = () => {
    onClose();
    // reset after transition finishes
    setTimeout(() => { setFpEmail(""); setSent(false); setSending(false); }, 300);
  };

  return (
    <Modal open={open} onClose={handleClose} title="Reset Password" width="max-w-sm">
      <div className="px-6 py-5">
        {!sent ? (
          <>
            <p className="text-sm text-slate-500 mb-5 leading-relaxed">
              Enter your account email and we will send you a link to reset your password.
            </p>
            <label className="text-sm font-semibold text-foreground block mb-1.5">Email Address</label>
            <div className="relative mb-5">
              <Mail size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
              <input
                type="email"
                value={fpEmail}
                onChange={e => setFpEmail(e.target.value)}
                onKeyDown={e => e.key === "Enter" && handleSend()}
                placeholder="you@hospital.org"
                className="w-full pl-9 pr-4 py-2.5 text-sm bg-slate-50 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
              />
            </div>
            <button
              onClick={handleSend}
              disabled={sending || !fpEmail.trim()}
              className="w-full bg-blue-600 hover:bg-blue-700 active:bg-blue-800 disabled:opacity-60 disabled:cursor-not-allowed text-white font-semibold py-2.5 rounded-xl transition-colors text-sm flex items-center justify-center gap-2 cursor-pointer"
            >
              {sending ? <><Spinner /> Sending…</> : "Send Reset Link"}
            </button>
          </>
        ) : (
          <div className="text-center py-4">
            <div className="w-12 h-12 rounded-full bg-emerald-100 flex items-center justify-center mx-auto mb-4">
              <CheckCircle size={22} className="text-emerald-600" />
            </div>
            <p className="font-semibold text-slate-900 mb-2">Check your inbox</p>
            <p className="text-sm text-slate-500 leading-relaxed mb-5">
              If an account exists for <span className="font-medium text-foreground">{fpEmail}</span>, a password reset link has been sent.
            </p>
            <button onClick={handleClose} className="text-sm text-blue-600 font-semibold hover:text-blue-700 active:text-blue-800 transition-colors cursor-pointer">
              ← Back to login
            </button>
          </div>
        )}
      </div>
    </Modal>
  );
}

// ─── Request Access Modal ────────────────────────────────────────────────────
function RequestAccessModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [form, setForm] = useState({ name: "", email: "", institution: "", reason: "" });
  const [submitted, setSubmitted] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const set = (k: string, v: string) => setForm(f => ({ ...f, [k]: v }));

  const handleSubmit = () => {
    if (!form.name.trim() || !form.email.trim() || !form.institution.trim()) {
      toast.error("Please fill in all required fields.");
      return;
    }
    setSubmitting(true);
    setTimeout(() => { setSubmitting(false); setSubmitted(true); }, 900);
  };

  const handleClose = () => {
    onClose();
    setTimeout(() => { setForm({ name: "", email: "", institution: "", reason: "" }); setSubmitted(false); setSubmitting(false); }, 300);
  };

  return (
    <Modal open={open} onClose={handleClose} title="Request Access" width="max-w-md">
      <div className="px-6 py-5">
        {!submitted ? (
          <div className="space-y-4">
            <p className="text-sm text-slate-500 leading-relaxed">
              Fill in your details and we will get back to you within 1–2 business days.
            </p>
            {[
              { key: "name", label: "Full Name", placeholder: "Dr. A. Smith", required: true },
              { key: "email", label: "Work Email", placeholder: "a.smith@hospital.org", required: true },
              { key: "institution", label: "Hospital / Institution", placeholder: "Your Hospital", required: true },
            ].map(({ key, label, placeholder, required }) => (
              <div key={key}>
                <label className="text-xs font-semibold text-foreground block mb-1.5">
                  {label} {required && <span className="text-red-500">*</span>}
                </label>
                <input
                  value={form[key as keyof typeof form]}
                  onChange={e => set(key, e.target.value)}
                  placeholder={placeholder}
                  className="w-full px-3 py-2.5 text-sm bg-slate-50 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all cursor-text"
                />
              </div>
            ))}
            <div>
              <label className="text-xs font-semibold text-foreground block mb-1.5">Reason for access</label>
              <textarea
                value={form.reason}
                onChange={e => set("reason", e.target.value)}
                placeholder="Briefly describe your use case…"
                rows={3}
                className="w-full px-3 py-2.5 text-sm bg-slate-50 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all resize-none cursor-text"
              />
            </div>
            <div className="flex gap-2 pt-1">
              <button onClick={handleClose} className="flex-1 border border-slate-200 text-sm font-medium py-2.5 rounded-xl hover:bg-slate-50 active:bg-slate-100 transition-colors cursor-pointer">
                Cancel
              </button>
              <button
                onClick={handleSubmit}
                disabled={submitting}
                className="flex-1 bg-blue-600 hover:bg-blue-700 active:bg-blue-800 disabled:opacity-60 disabled:cursor-not-allowed text-white text-sm font-semibold py-2.5 rounded-xl transition-colors flex items-center justify-center gap-2 cursor-pointer"
              >
                {submitting
                  ? <><Spinner /> Submitting…</>
                  : "Submit Request"}
              </button>
            </div>
          </div>
        ) : (
          <div className="text-center py-6">
            <div className="w-12 h-12 rounded-full bg-emerald-100 flex items-center justify-center mx-auto mb-4">
              <CheckCircle size={22} className="text-emerald-600" />
            </div>
            <p className="font-semibold text-slate-900 mb-2">Request submitted!</p>
            <p className="text-sm text-slate-500 leading-relaxed mb-5">
              {"We've received your request for "}
              <span className="font-medium text-foreground">{form.email}</span>
              {". Expect a reply within 1–2 business days."}
            </p>
            <button onClick={handleClose} className="text-sm text-blue-600 font-semibold hover:text-blue-700 active:text-blue-800 transition-colors cursor-pointer">
              ← Back to login
            </button>
          </div>
        )}
      </div>
    </Modal>
  );
}

// ─── Help Modal ───────────────────────────────────────────────────────────────
function HelpModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <Modal open={open} onClose={onClose} title="Help & Support" width="max-w-sm">
      <div className="px-6 py-5 space-y-4">
        <div className="bg-blue-50 border border-blue-100 rounded-xl p-4">
          <p className="text-xs font-bold text-blue-700 uppercase tracking-wide mb-2">Demo Project</p>
          <p className="text-sm text-blue-800 leading-relaxed">
            This is an academic AI healthcare demo. Use the credentials shown on the login page to explore the dashboard.
          </p>
        </div>
        {[
          { icon: Mail, label: "Email Support", value: "support@retinasense.ai", action: () => toast.info("Email copied to clipboard.") },
          { icon: MessageSquare, label: "GitHub Issues", value: "github.com/retinasense/ai", action: () => toast.info("Opening GitHub…") },
          { icon: FileText, label: "Documentation", value: "View project README", action: () => toast.info("Opening docs…") },
        ].map(({ icon: Icon, label, value, action }) => (
          <button key={label} onClick={action}
            className="w-full flex items-center gap-3 p-3 rounded-xl hover:bg-slate-50 active:bg-slate-100 transition-colors text-left cursor-pointer">
            <div className="w-8 h-8 rounded-lg bg-slate-100 flex items-center justify-center flex-shrink-0">
              <Icon size={14} className="text-slate-600" />
            </div>
            <div>
              <p className="text-xs font-semibold text-slate-800">{label}</p>
              <p className="text-xs text-slate-500">{value}</p>
            </div>
            <ChevronRight size={14} className="text-slate-400 ml-auto flex-shrink-0" />
          </button>
        ))}
        <div className="border-t border-slate-100 pt-3">
          <p className="text-xs text-slate-400 text-center">RetinaSense AI · Academic Project 2024 · Deep Learning + CV</p>
        </div>
      </div>
    </Modal>
  );
}

const SHAKE_STYLE = `
  @keyframes shake { 0%,100%{transform:translateX(0)} 10%,30%,50%,70%,90%{transform:translateX(-4px)} 20%,40%,60%,80%{transform:translateX(4px)} }
  .rs-shake { animation: shake .4s ease-in-out; }
`;

function LoginPage({ onNav, onLogin }: { onNav: (p: Page) => void; onLogin: (email: string, password: string) => Promise<boolean> }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPass, setShowPass] = useState(false);
  const [remember, setRemember] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [shake, setShake] = useState(false);
  const [showForgot, setShowForgot] = useState(false);

  const triggerShake = () => {
    setShake(false);
    requestAnimationFrame(() => requestAnimationFrame(() => setShake(true)));
    setTimeout(() => setShake(false), 500);
  };

  const handleSignIn = async () => {
    setError("");
    if (!email.trim() || !password) {
      setError("Email and password are required.");
      triggerShake();
      return;
    }
    setLoading(true);
    const ok = await onLogin(email.trim(), password);
    setLoading(false);
    if (!ok) {
      setError("Invalid email or password. Please try again.");
      setPassword("");
      triggerShake();
    }
  };

  const inputBase = "w-full pl-9 pr-4 py-2.5 text-sm bg-slate-50 border rounded-xl focus:outline-none focus:ring-2 focus:border-transparent transition-all cursor-text";
  const inputNormal = "border-slate-200 focus:ring-blue-500";
  const inputError = "border-red-300 bg-red-50 focus:ring-red-400";

  return (
    <>
      <style>{SHAKE_STYLE}</style>
      <ForgotPasswordModal open={showForgot} onClose={() => setShowForgot(false)} />

      <div className="min-h-screen grid lg:grid-cols-2" style={{ fontFamily: "Inter, sans-serif" }}>
        {/* Left panel */}
        <div className="relative hidden lg:flex flex-col bg-gradient-to-br from-slate-900 via-blue-950 to-indigo-950 p-10 overflow-hidden">
          <div className="absolute inset-0 overflow-hidden">
            <div className="absolute top-1/4 left-1/3 w-96 h-96 rounded-full bg-blue-600/20 blur-3xl" />
            <div className="absolute bottom-1/4 right-1/4 w-64 h-64 rounded-full bg-indigo-500/20 blur-3xl" />
          </div>
          <div className="relative flex items-center gap-2.5 mb-auto">
            <button
              onClick={() => onNav("dashboard")}
              className="flex items-center gap-2.5 cursor-pointer group"
            >
              <div className="w-9 h-9 rounded-xl bg-white/10 backdrop-blur-sm border border-white/20 flex items-center justify-center group-hover:bg-white/20 transition-colors">
                <Eye size={17} className="text-white" />
              </div>
              <span className="font-bold text-white text-lg group-hover:text-white/80 transition-colors">RetinaSense AI</span>
            </button>
          </div>
          <div className="relative flex-1 flex items-center justify-center py-12">
            <div className="relative w-full max-w-sm">
              <svg viewBox="0 0 400 400" className="w-full h-auto drop-shadow-2xl">
                <defs>
                  <radialGradient id="eyeGlow" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stopColor="#0ea5e9" stopOpacity="0.3"/>
                    <stop offset="100%" stopColor="#0ea5e9" stopOpacity="0"/>
                  </radialGradient>
                  <linearGradient id="irisGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#06b6d4"/>
                    <stop offset="50%" stopColor="#0891b2"/>
                    <stop offset="100%" stopColor="#0e7490"/>
                  </linearGradient>
                  <filter id="glow">
                    <feGaussianBlur stdDeviation="3" result="blur"/>
                    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                  </filter>
                </defs>
                {/* Background glow */}
                <circle cx="200" cy="200" r="180" fill="url(#eyeGlow)"/>
                {/* Eye outline */}
                <ellipse cx="200" cy="200" rx="160" ry="90" fill="none" stroke="#0ea5e9" strokeWidth="2" opacity="0.6"/>
                <ellipse cx="200" cy="200" rx="160" ry="90" fill="none" stroke="#06b6d4" strokeWidth="1" opacity="0.3" strokeDasharray="8 4"/>
                {/* Sclera */}
                <ellipse cx="200" cy="200" rx="155" ry="85" fill="#f8fafc" opacity="0.95"/>
                {/* Iris */}
                <circle cx="200" cy="200" r="65" fill="url(#irisGrad)"/>
                <circle cx="200" cy="200" r="65" fill="none" stroke="#22d3ee" strokeWidth="2" opacity="0.5"/>
                {/* Iris detail rings */}
                <circle cx="200" cy="200" r="55" fill="none" stroke="#67e8f9" strokeWidth="0.5" opacity="0.4"/>
                <circle cx="200" cy="200" r="45" fill="none" stroke="#67e8f9" strokeWidth="0.5" opacity="0.3"/>
                {/* Pupil */}
                <circle cx="200" cy="200" r="28" fill="#0f172a"/>
                <circle cx="200" cy="200" r="28" fill="none" stroke="#0ea5e9" strokeWidth="1" opacity="0.4"/>
                {/* Pupil highlight */}
                <circle cx="190" cy="190" r="8" fill="white" opacity="0.7"/>
                <circle cx="210" cy="195" r="4" fill="white" opacity="0.4"/>
                {/* Scan lines */}
                <line x1="40" y1="200" x2="360" y2="200" stroke="#22d3ee" strokeWidth="1.5" opacity="0.6" filter="url(#glow)">
                  <animate attributeName="y1" values="100;300;100" dur="4s" repeatCount="indefinite"/>
                  <animate attributeName="y2" values="100;300;100" dur="4s" repeatCount="indefinite"/>
                </line>
                {/* Corner brackets */}
                <path d="M60 120 L60 80 L100 80" fill="none" stroke="#22d3ee" strokeWidth="2" opacity="0.7"/>
                <path d="M340 120 L340 80 L300 80" fill="none" stroke="#22d3ee" strokeWidth="2" opacity="0.7"/>
                <path d="M60 280 L60 320 L100 320" fill="none" stroke="#22d3ee" strokeWidth="2" opacity="0.7"/>
                <path d="M340 280 L340 320 L300 320" fill="none" stroke="#22d3ee" strokeWidth="2" opacity="0.7"/>
                {/* Target circles */}
                <circle cx="200" cy="200" r="80" fill="none" stroke="#22d3ee" strokeWidth="0.5" opacity="0.3" strokeDasharray="4 8"/>
                <circle cx="200" cy="200" r="100" fill="none" stroke="#22d3ee" strokeWidth="0.5" opacity="0.2" strokeDasharray="2 6"/>
                {/* Scan dots */}
                <circle cx="150" cy="180" r="3" fill="#22d3ee" opacity="0.6">
                  <animate attributeName="opacity" values="0.6;0.1;0.6" dur="2s" repeatCount="indefinite"/>
                </circle>
                <circle cx="250" cy="220" r="3" fill="#22d3ee" opacity="0.4">
                  <animate attributeName="opacity" values="0.4;0.8;0.4" dur="1.5s" repeatCount="indefinite"/>
                </circle>
                <circle cx="200" cy="150" r="2" fill="#22d3ee" opacity="0.5">
                  <animate attributeName="opacity" values="0.5;0.1;0.5" dur="2.5s" repeatCount="indefinite"/>
                </circle>
              </svg>
            </div>
          </div>
        </div>

        {/* Right panel */}
        <div className="flex items-center justify-center p-8 bg-white relative">
          <div className="w-full max-w-sm">
            {/* Mobile logo */}
            <div className="flex items-center gap-2 mb-8 lg:hidden">
              <button onClick={() => onNav("dashboard")} className="flex items-center gap-2 cursor-pointer group">
                <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center group-hover:opacity-80 transition-opacity">
                  <Eye size={15} className="text-white" />
                </div>
                <span className="font-bold text-slate-900 group-hover:text-foreground transition-colors">RetinaSense AI</span>
              </button>
            </div>

            <h1 className="text-2xl font-extrabold text-slate-900 mb-1">Welcome back</h1>
            <p className="text-sm text-slate-500 mb-6">Sign in to access your clinical dashboard</p>

            {/* Error banner */}
            {error && (
              <div className="flex items-start gap-2.5 bg-red-50 border border-red-200 rounded-xl px-4 py-3 mb-4">
                <XCircle size={15} className="text-red-500 flex-shrink-0 mt-0.5" />
                <p className="text-sm text-red-700 leading-snug">{error}</p>
              </div>
            )}

            {/* Form */}
            <div className={cn("space-y-4", shake && "rs-shake")}>
              {/* Email */}
              <div>
                <label className="text-sm font-semibold text-foreground block mb-1.5 cursor-default">Email Address</label>
                <div className="relative">
                  <Mail size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" />
                  <input
                    type="email"
                    value={email}
                    onChange={e => { setEmail(e.target.value); setError(""); }}
                    onKeyDown={e => e.key === "Enter" && handleSignIn()}
                    disabled={loading}
                    placeholder="you@hospital.org"
                    className={cn(inputBase, error ? inputError : inputNormal, loading && "opacity-60 cursor-not-allowed")}
                  />
                </div>
              </div>

              {/* Password */}
              <div>
                <label className="text-sm font-semibold text-foreground block mb-1.5 cursor-default">Password</label>
                <div className="relative">
                  <Lock size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" />
                  <input
                    type={showPass ? "text" : "password"}
                    value={password}
                    onChange={e => { setPassword(e.target.value); setError(""); }}
                    onKeyDown={e => e.key === "Enter" && handleSignIn()}
                    disabled={loading}
                    placeholder="••••••••••"
                    className={cn(inputBase, "pr-10", error ? inputError : inputNormal, loading && "opacity-60 cursor-not-allowed")}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPass(v => !v)}
                    disabled={loading}
                    title={showPass ? "Hide password" : "Show password"}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 active:text-slate-800 disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer transition-colors p-0.5 rounded"
                  >
                    {showPass ? <X size={14} /> : <Eye size={14} />}
                  </button>
                </div>
              </div>

              {/* Remember me + Forgot password */}
              <div className="flex items-center justify-between">
                <label className="flex items-center gap-2 cursor-pointer select-none group">
                  <input
                    type="checkbox"
                    checked={remember}
                    onChange={e => setRemember(e.target.checked)}
                    disabled={loading}
                    className="w-4 h-4 rounded border-slate-300 text-blue-600 cursor-pointer disabled:cursor-not-allowed accent-blue-600"
                  />
                  <span className="text-sm text-slate-600 group-hover:text-slate-800 transition-colors">Remember me</span>
                </label>
                <button
                  type="button"
                  onClick={() => setShowForgot(true)}
                  disabled={loading}
                  className="text-sm text-blue-600 hover:text-blue-700 active:text-blue-800 disabled:opacity-40 disabled:cursor-not-allowed font-medium transition-colors cursor-pointer"
                >
                  Forgot password?
                </button>
              </div>

              {/* Login + Signup buttons */}
              <div className="grid grid-cols-2 gap-3">
                <button
                  onClick={handleSignIn}
                  disabled={loading}
                  className="bg-blue-600 hover:bg-blue-700 active:bg-blue-800 active:scale-[0.98] disabled:opacity-70 disabled:cursor-not-allowed text-white font-semibold py-2.5 rounded-xl shadow-sm transition-colors duration-200 text-sm flex items-center justify-center gap-2 cursor-pointer"
                >
                  {loading ? <><Spinner /> Signing in…</> : "Login"}
                </button>
                <button
                  onClick={() => onNav("signup")}
                  disabled={loading}
                  className="border border-slate-200 hover:bg-slate-50 active:bg-slate-100 active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed text-foreground font-semibold py-2.5 rounded-xl shadow-sm transition-colors duration-200 text-sm flex items-center justify-center gap-2 cursor-pointer"
                >
                  Sign up
                </button>
              </div>
            </div>

          </div>
        </div>
      </div>
    </>
  );
}

// ─── Page: Sign Up ────────────────────────────────────────────────────────────
function SignUpPage({ onNav }: { onNav: (p: Page) => void }) {
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPass, setShowPass] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);
  const [shake, setShake] = useState(false);

  const triggerShake = () => {
    setShake(false);
    requestAnimationFrame(() => requestAnimationFrame(() => setShake(true)));
    setTimeout(() => setShake(false), 500);
  };

  const handleSignUp = async () => {
    setError("");
    if (!firstName.trim() || !lastName.trim() || !email.trim() || !password || !confirmPassword) {
      setError("All fields are required.");
      triggerShake();
      return;
    }
    if (password.length < 6) {
      setError("Password must be at least 6 characters.");
      triggerShake();
      return;
    }
    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      triggerShake();
      return;
    }
    setLoading(true);
    try {
      const { authApi } = await import("../services/api");
      await authApi.register({ first_name: firstName.trim(), last_name: lastName.trim(), email: email.trim(), password });
      setSuccess(true);
    } catch (e: any) {
      setError(e.message || "Registration failed. Please try again.");
      triggerShake();
    } finally {
      setLoading(false);
    }
  };

  const inputBase = "w-full pl-9 pr-4 py-2.5 text-sm bg-slate-50 border rounded-xl focus:outline-none focus:ring-2 focus:border-transparent transition-all cursor-text";
  const inputNormal = "border-slate-200 focus:ring-blue-500";
  const inputError = "border-red-300 bg-red-50 focus:ring-red-400";

  if (success) {
    return (
      <div className="min-h-screen grid lg:grid-cols-2" style={{ fontFamily: "Inter, sans-serif" }}>
        <div className="relative hidden lg:flex flex-col bg-gradient-to-br from-slate-900 via-blue-950 to-indigo-950 p-10 overflow-hidden">
          <div className="absolute inset-0 overflow-hidden">
            <div className="absolute top-1/4 left-1/3 w-96 h-96 rounded-full bg-blue-600/20 blur-3xl" />
            <div className="absolute bottom-1/4 right-1/4 w-64 h-64 rounded-full bg-indigo-500/20 blur-3xl" />
          </div>
          <div className="relative flex items-center gap-2.5 mb-auto">
            <div className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-xl bg-white/10 backdrop-blur-sm border border-white/20 flex items-center justify-center">
                <Eye size={17} className="text-white" />
              </div>
              <span className="font-bold text-white text-lg">RetinaSense AI</span>
            </div>
          </div>
          <div className="relative flex-1 flex items-center justify-center py-12">
            <div className="relative w-full max-w-sm">
              <svg viewBox="0 0 400 400" className="w-full h-auto drop-shadow-2xl">
                <defs>
                  <radialGradient id="eyeGlow2" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stopColor="#10b981" stopOpacity="0.3"/>
                    <stop offset="100%" stopColor="#10b981" stopOpacity="0"/>
                  </radialGradient>
                  <linearGradient id="irisGrad2" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#34d399"/>
                    <stop offset="100%" stopColor="#059669"/>
                  </linearGradient>
                </defs>
                <circle cx="200" cy="200" r="180" fill="url(#eyeGlow2)"/>
                <ellipse cx="200" cy="200" rx="160" ry="90" fill="none" stroke="#10b981" strokeWidth="2" opacity="0.6"/>
                <ellipse cx="200" cy="200" rx="155" ry="85" fill="#f8fafc" opacity="0.95"/>
                <circle cx="200" cy="200" r="65" fill="url(#irisGrad2)"/>
                <circle cx="200" cy="200" r="28" fill="#0f172a"/>
                <circle cx="190" cy="190" r="8" fill="white" opacity="0.7"/>
                <path d="M60 120 L60 80 L100 80" fill="none" stroke="#10b981" strokeWidth="2" opacity="0.7"/>
                <path d="M340 120 L340 80 L300 80" fill="none" stroke="#10b981" strokeWidth="2" opacity="0.7"/>
                <path d="M60 280 L60 320 L100 320" fill="none" stroke="#10b981" strokeWidth="2" opacity="0.7"/>
                <path d="M340 280 L340 320 L300 320" fill="none" stroke="#10b981" strokeWidth="2" opacity="0.7"/>
              </svg>
            </div>
          </div>
        </div>
        <div className="flex items-center justify-center p-8 bg-white">
          <div className="w-full max-w-sm text-center">
            <div className="w-16 h-16 rounded-full bg-emerald-100 flex items-center justify-center mx-auto mb-5">
              <CheckCircle size={32} className="text-emerald-600" />
            </div>
            <h1 className="text-2xl font-extrabold text-slate-900 mb-2">Account created!</h1>
            <p className="text-sm text-slate-500 mb-6">You can now sign in with your new account.</p>
            <button
              onClick={() => onNav("login")}
              className="w-full bg-blue-600 hover:bg-blue-700 active:bg-blue-800 text-white font-semibold py-2.5 rounded-xl shadow-sm transition-colors duration-200 text-sm cursor-pointer"
            >
              Go to Sign In
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <>
      <style>{SHAKE_STYLE}</style>
      <div className="min-h-screen grid lg:grid-cols-2" style={{ fontFamily: "Inter, sans-serif" }}>
        <div className="relative hidden lg:flex flex-col bg-gradient-to-br from-slate-900 via-blue-950 to-indigo-950 p-10 overflow-hidden">
          <div className="absolute inset-0 overflow-hidden">
            <div className="absolute top-1/4 left-1/3 w-96 h-96 rounded-full bg-blue-600/20 blur-3xl" />
            <div className="absolute bottom-1/4 right-1/4 w-64 h-64 rounded-full bg-indigo-500/20 blur-3xl" />
          </div>
          <div className="relative flex items-center gap-2.5 mb-auto">
            <div className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-xl bg-white/10 backdrop-blur-sm border border-white/20 flex items-center justify-center">
                <Eye size={17} className="text-white" />
              </div>
              <span className="font-bold text-white text-lg">RetinaSense AI</span>
            </div>
          </div>
          <div className="relative flex-1 flex items-center justify-center py-12">
            <div className="relative w-full max-w-sm">
              <svg viewBox="0 0 400 400" className="w-full h-auto drop-shadow-2xl">
                <defs>
                  <radialGradient id="eyeGlow3" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stopColor="#0ea5e9" stopOpacity="0.3"/>
                    <stop offset="100%" stopColor="#0ea5e9" stopOpacity="0"/>
                  </radialGradient>
                  <linearGradient id="irisGrad3" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#06b6d4"/>
                    <stop offset="50%" stopColor="#0891b2"/>
                    <stop offset="100%" stopColor="#0e7490"/>
                  </linearGradient>
                  <filter id="glow3">
                    <feGaussianBlur stdDeviation="3" result="blur"/>
                    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                  </filter>
                </defs>
                <circle cx="200" cy="200" r="180" fill="url(#eyeGlow3)"/>
                <ellipse cx="200" cy="200" rx="160" ry="90" fill="none" stroke="#0ea5e9" strokeWidth="2" opacity="0.6"/>
                <ellipse cx="200" cy="200" rx="155" ry="85" fill="#f8fafc" opacity="0.95"/>
                <circle cx="200" cy="200" r="65" fill="url(#irisGrad3)"/>
                <circle cx="200" cy="200" r="55" fill="none" stroke="#67e8f9" strokeWidth="0.5" opacity="0.4"/>
                <circle cx="200" cy="200" r="45" fill="none" stroke="#67e8f9" strokeWidth="0.5" opacity="0.3"/>
                <circle cx="200" cy="200" r="28" fill="#0f172a"/>
                <circle cx="190" cy="190" r="8" fill="white" opacity="0.7"/>
                <circle cx="210" cy="195" r="4" fill="white" opacity="0.4"/>
                <line x1="40" y1="200" x2="360" y2="200" stroke="#22d3ee" strokeWidth="1.5" opacity="0.6" filter="url(#glow3)">
                  <animate attributeName="y1" values="100;300;100" dur="4s" repeatCount="indefinite"/>
                  <animate attributeName="y2" values="100;300;100" dur="4s" repeatCount="indefinite"/>
                </line>
                <path d="M60 120 L60 80 L100 80" fill="none" stroke="#22d3ee" strokeWidth="2" opacity="0.7"/>
                <path d="M340 120 L340 80 L300 80" fill="none" stroke="#22d3ee" strokeWidth="2" opacity="0.7"/>
                <path d="M60 280 L60 320 L100 320" fill="none" stroke="#22d3ee" strokeWidth="2" opacity="0.7"/>
                <path d="M340 280 L340 320 L300 320" fill="none" stroke="#22d3ee" strokeWidth="2" opacity="0.7"/>
                <circle cx="200" cy="200" r="80" fill="none" stroke="#22d3ee" strokeWidth="0.5" opacity="0.3" strokeDasharray="4 8"/>
              </svg>
            </div>
          </div>
        </div>

        <div className="flex items-center justify-center p-8 bg-white relative">
          <div className="w-full max-w-sm">
            <div className="flex items-center gap-2 mb-8 lg:hidden">
              <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center">
                <Eye size={15} className="text-white" />
              </div>
              <span className="font-bold text-slate-900">RetinaSense AI</span>
            </div>

            <h1 className="text-2xl font-extrabold text-slate-900 mb-1">Create account</h1>
            <p className="text-sm text-slate-500 mb-6">Sign up to start using RetinaSense AI</p>

            {error && (
              <div className="flex items-start gap-2.5 bg-red-50 border border-red-200 rounded-xl px-4 py-3 mb-4">
                <XCircle size={15} className="text-red-500 flex-shrink-0 mt-0.5" />
                <p className="text-sm text-red-700 leading-snug">{error}</p>
              </div>
            )}

            <div className={cn("space-y-4", shake && "rs-shake")}>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-sm font-semibold text-foreground block mb-1.5">First Name</label>
                  <div className="relative">
                    <User size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" />
                    <input
                      type="text"
                      value={firstName}
                      onChange={e => { setFirstName(e.target.value); setError(""); }}
                      disabled={loading}
                      placeholder="John"
                      className={cn(inputBase, inputNormal, loading && "opacity-60 cursor-not-allowed")}
                    />
                  </div>
                </div>
                <div>
                  <label className="text-sm font-semibold text-foreground block mb-1.5">Last Name</label>
                  <div className="relative">
                    <User size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" />
                    <input
                      type="text"
                      value={lastName}
                      onChange={e => { setLastName(e.target.value); setError(""); }}
                      disabled={loading}
                      placeholder="Doe"
                      className={cn(inputBase, inputNormal, loading && "opacity-60 cursor-not-allowed")}
                    />
                  </div>
                </div>
              </div>

              <div>
                <label className="text-sm font-semibold text-foreground block mb-1.5">Email Address</label>
                <div className="relative">
                  <Mail size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" />
                  <input
                    type="email"
                    value={email}
                    onChange={e => { setEmail(e.target.value); setError(""); }}
                    onKeyDown={e => e.key === "Enter" && handleSignUp()}
                    disabled={loading}
                    placeholder="you@hospital.org"
                    className={cn(inputBase, inputNormal, loading && "opacity-60 cursor-not-allowed")}
                  />
                </div>
              </div>

              <div>
                <label className="text-sm font-semibold text-foreground block mb-1.5">Password</label>
                <div className="relative">
                  <Lock size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" />
                  <input
                    type={showPass ? "text" : "password"}
                    value={password}
                    onChange={e => { setPassword(e.target.value); setError(""); }}
                    disabled={loading}
                    placeholder="At least 6 characters"
                    className={cn(inputBase, "pr-10", inputNormal, loading && "opacity-60 cursor-not-allowed")}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPass(v => !v)}
                    disabled={loading}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer transition-colors p-0.5 rounded"
                  >
                    {showPass ? <X size={14} /> : <Eye size={14} />}
                  </button>
                </div>
              </div>

              <div>
                <label className="text-sm font-semibold text-foreground block mb-1.5">Confirm Password</label>
                <div className="relative">
                  <Lock size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" />
                  <input
                    type={showPass ? "text" : "password"}
                    value={confirmPassword}
                    onChange={e => { setConfirmPassword(e.target.value); setError(""); }}
                    onKeyDown={e => e.key === "Enter" && handleSignUp()}
                    disabled={loading}
                    placeholder="Re-enter password"
                    className={cn(inputBase, inputNormal, loading && "opacity-60 cursor-not-allowed")}
                  />
                </div>
              </div>

              <button
                onClick={handleSignUp}
                disabled={loading}
                className="w-full bg-blue-600 hover:bg-blue-700 active:bg-blue-800 active:scale-[0.98] disabled:opacity-70 disabled:cursor-not-allowed text-white font-semibold py-2.5 rounded-xl shadow-sm transition-colors duration-200 text-sm flex items-center justify-center gap-2 cursor-pointer"
              >
                {loading ? <><Spinner /> Creating account…</> : "Create Account"}
              </button>
            </div>

            <p className="text-center text-sm text-slate-500 mt-6">
              Already have an account?{" "}
              <button
                onClick={() => onNav("login")}
                className="text-blue-600 font-semibold hover:text-blue-700 active:text-blue-800 transition-colors cursor-pointer"
              >
                Sign in
              </button>
            </p>

            <div className="flex justify-center mt-5">
              <button
                onClick={() => onNav("login")}
                className="flex items-center gap-1.5 text-sm text-slate-400 hover:text-slate-600 active:text-slate-800 transition-colors cursor-pointer"
              >
                <ChevronRight size={14} className="rotate-180" />
                Back to Sign In
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

// ─── Page: Dashboard ──────────────────────────────────────────────────────────
function DashboardPage({ onNav }: { onNav: (p: Page) => void }) {
  const dashboard = useDashboard();
  const kpis = dashboard.kpis?.map((k: any) => ({
    title: k.title, value: k.value, change: k.change,
    icon: k.title === "Total Scans" ? ScanEye : k.title.includes("Accuracy") ? Target : k.title.includes("Patient") ? Users : FileText,
    color: k.title === "Total Scans" ? "bg-blue-600" : k.title.includes("Accuracy") ? "bg-indigo-600" : k.title.includes("Patient") ? "bg-emerald-600" : "bg-amber-500",
    nav: k.title === "Total Scans" ? "scan" as Page : k.title.includes("Accuracy") ? "dashboard" as Page : k.title.includes("Patient") ? "patients" as Page : "reports" as Page,
  })) ?? [
    { title: "Total Scans", value: "—", change: "—", icon: ScanEye, color: "bg-blue-600", nav: "scan" as Page },
    { title: "AI Accuracy", value: "—", change: "—", icon: Target, color: "bg-indigo-600", nav: "dashboard" as Page },
    { title: "Active Patients", value: "—", change: "—", icon: Users, color: "bg-emerald-600", nav: "patients" as Page },
    { title: "Reports Generated", value: "—", change: "—", icon: FileText, color: "bg-amber-500", nav: "reports" as Page },
  ];
  const activityData = dashboard.scanActivity ?? [];
  const perfData = dashboard.aiPerformance ?? [];
  const activity = dashboard.recentActivity ?? [];

  // Disease distribution: normalize values to sum to 100% and assign distinct,
  // non-brand colors so each segment is clearly readable.
  const DIST_COLORS: Record<string, string> = {
    "Diabetic Retinopathy": "#F59E0B",
    "Glaucoma": "#8B5CF6",
    "AMD": "#06B6D4",
    "Macular Edema": "#F97316",
    "Hypertensive Retinopathy": "#EC4899",
    "Healthy": "#10B981",
  };
  const distData = useMemo(() => {
    const raw = (dashboard.diseaseDistribution ?? []).map((d: any) => ({ ...d }));
    const total = raw.reduce((s: number, d: any) => s + (Number(d.value) || 0), 0);
    if (total > 0 && Math.abs(total - 100) > 0.05) {
      raw.forEach((d: any) => { d.value = Number((((Number(d.value) || 0) / total) * 100).toFixed(1)); });
    }
    raw.forEach((d: any) => { d.color = DIST_COLORS[d.name] || d.color || "#94A3B8"; });
    return raw;
  }, [dashboard.diseaseDistribution]);

  useEffect(() => {
    if (dashboard.error) toast.error("Failed to load dashboard data");
  }, [dashboard.error]);

  return (
    <div className="space-y-6">
      {/* KPIs */}
      {dashboard.loading && !dashboard.kpis ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {Array.from({ length: 4 }).map((_, i) => <SkeletonCard key={i} lines={2} />)}
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {kpis.map(kpi => <KpiCard key={kpi.title} {...kpi} onClick={() => onNav(kpi.nav)} />)}
        </div>
      )}

      <div className="grid lg:grid-cols-3 gap-4">
        {/* Scan Activity */}
        <div className="lg:col-span-2 bg-card rounded-2xl border border-border p-5 shadow-sm">
          <div className="flex items-center justify-between mb-5">
            <div>
              <h3 className="font-semibold text-foreground text-sm">Weekly Scan Activity</h3>
              <p className="text-xs text-muted-foreground">Scans processed vs analyzed</p>
            </div>
            <div className="flex gap-4 text-xs">
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-blue-500" />Scans</span>
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-indigo-400" />Analyzed</span>
            </div>
          </div>
          {dashboard.loading && activityData.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 space-y-3">
            <Skeleton className="h-5 w-1/4" />
            <Skeleton className="h-48 w-full" />
          </div>
          ) : activityData.length > 0 ? (
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={activityData} barSize={14} barGap={4}>
              <CartesianGrid key="grid" strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
              <XAxis key="xaxis" dataKey="day" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: "#94a3b8" }} />
              <YAxis key="yaxis" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: "#94a3b8" }} />
              <Tooltip key="tooltip" contentStyle={{ borderRadius: "12px", border: "1px solid #e2e8f0", fontSize: 12 }} />
              <Bar key="bar-scans" dataKey="scans" name="Scans" fill="#2563EB" radius={[4, 4, 0, 0]} />
              <Bar key="bar-analyzed" dataKey="analyzed" name="Analyzed" fill="#818cf8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          ) : (
          <div className="flex items-center justify-center h-64 text-muted-foreground text-sm">
            No scan activity yet. Complete a scan to see weekly data.
          </div>
          )}
        </div>

        {/* Disease Distribution */}
        <div className="bg-card rounded-2xl border border-border p-5 shadow-sm">
          <div className="mb-5">
            <h3 className="font-semibold text-foreground text-sm">Disease Distribution</h3>
            <p className="text-xs text-muted-foreground">This month's detections</p>
          </div>
          {dashboard.loading && distData.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 space-y-3">
            <Skeleton className="h-5 w-1/4" />
            <Skeleton className="h-40 w-full" />
          </div>
          ) : distData.length > 0 ? (
          <>
          <ResponsiveContainer width="100%" height={160}>
            <PieChart>
              <Pie data={distData} cx="50%" cy="50%" innerRadius={45} outerRadius={70} dataKey="value" paddingAngle={3}>
                {distData.map((d: any, i: number) => <Cell key={`cell-${i}`} fill={d.color} />)}
              </Pie>
              <Tooltip contentStyle={{ borderRadius: "12px", border: "1px solid #e2e8f0", fontSize: 11 }} />
            </PieChart>
          </ResponsiveContainer>
          <div className="mt-3 space-y-1.5">
            {distData.map((d: any) => (
              <div key={d.name} className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: d.color }} />
                <span className="text-xs text-muted-foreground flex-1 truncate">{d.name}</span>
                <span className="text-xs font-semibold text-foreground">{d.value}%</span>
              </div>
            ))}
          </div>
          </>
          ) : (
          <div className="flex items-center justify-center h-64 text-muted-foreground text-sm">
            No disease distribution data yet.
          </div>
          )}
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-4">
        {/* AI Performance */}
        <div className="lg:col-span-2 bg-card rounded-2xl border border-border p-5 shadow-sm">
          <div className="flex items-center justify-between mb-5">
            <div>
              <h3 className="font-semibold text-foreground text-sm">AI Model Performance</h3>
              <p className="text-xs text-muted-foreground">Accuracy & F1 score trend</p>
            </div>
            <Badge variant="success">{perfData.length ? `${perfData[perfData.length - 1].accuracy}%` : "—"} Today</Badge>
          </div>
          {/* Gradient defs live outside recharts to avoid duplicate-key warnings */}
          <svg width="0" height="0" style={{ position: "absolute" }}>
            <defs>
              <linearGradient id="gAcc" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#2563EB" stopOpacity={0.15} />
                <stop offset="100%" stopColor="#2563EB" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="gF1" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#4F46E5" stopOpacity={0.1} />
                <stop offset="100%" stopColor="#4F46E5" stopOpacity={0} />
              </linearGradient>
            </defs>
          </svg>
          {dashboard.loading && perfData.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 space-y-3">
            <Skeleton className="h-5 w-1/4" />
            <Skeleton className="h-44 w-full" />
          </div>
          ) : perfData.length > 0 ? (
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart data={perfData}>
              <CartesianGrid key="grid" strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
              <XAxis key="xaxis" dataKey="month" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: "#94a3b8" }} />
              <YAxis key="yaxis" domain={[90, 100]} axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: "#94a3b8" }} />
              <Tooltip key="tooltip" contentStyle={{ borderRadius: "12px", border: "1px solid #e2e8f0", fontSize: 12 }} />
              <Area key="area-accuracy" type="monotone" dataKey="accuracy" name="Accuracy" stroke="#2563EB" strokeWidth={2} fill="url(#gAcc)" />
              <Area key="area-f1" type="monotone" dataKey="f1" name="F1 Score" stroke="#4F46E5" strokeWidth={2} fill="url(#gF1)" strokeDasharray="4 2" />
            </AreaChart>
          </ResponsiveContainer>
          ) : (
          <div className="flex items-center justify-center h-64 text-muted-foreground text-sm">
            No performance data yet. Run an analysis to see model metrics.
          </div>
          )}
        </div>

        {/* Recent Activity */}
        <div className="bg-card rounded-2xl border border-border p-5 shadow-sm">
          <div className="flex items-center justify-between mb-5">
            <h3 className="font-semibold text-foreground text-sm">Recent Activity</h3>
            <button onClick={() => onNav("reports")} className="text-xs text-blue-600 font-medium hover:text-blue-700 transition-colors">View all →</button>
          </div>
          <div className="space-y-3">
            {dashboard.loading && activity.length === 0 ? (
              <div className="space-y-3">
                {Array.from({ length: 3 }).map((_, i) => (
                  <div key={i} className="flex items-start gap-3 rounded-xl px-2 py-1.5">
                    <Skeleton className="w-6 h-6 rounded-lg" />
                    <div className="flex-1 space-y-2">
                      <Skeleton className="h-3 w-2/3" />
                      <Skeleton className="h-3 w-1/2" />
                    </div>
                  </div>
                ))}
              </div>
            ) : activity.length > 0 ? activity.map(({ time, action, patient, result, type }: any, i: number) => {
              const icons = { warning: <AlertTriangle size={13} className="text-amber-500" />, info: <Activity size={13} className="text-blue-500" />, success: <CheckCircle size={13} className="text-emerald-500" />, error: <XCircle size={13} className="text-red-500" /> };
              return (
                <div
                  key={i}
                  className="flex items-start gap-3 rounded-xl px-2 py-1.5 -mx-2 hover:bg-muted/50 transition-colors cursor-default"
                >
                  <div className="w-6 h-6 rounded-lg bg-muted flex items-center justify-center flex-shrink-0 mt-0.5">
                    {icons[type as keyof typeof icons]}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-semibold text-foreground">{action}</p>
                    <p className="text-xs text-muted-foreground truncate">{toTitleCase(patient)} — {result}</p>
                  </div>
                  <span className="text-xs text-muted-foreground flex-shrink-0">{time}</span>
                </div>
              );
            }) : !dashboard.loading ? (
              <div className="text-center py-8 text-muted-foreground text-sm">
                No recent activity yet.
              </div>
            ) : null}
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-gradient-to-r from-blue-600 to-indigo-600 rounded-2xl p-5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h3 className="font-semibold text-white">Ready to analyze a new scan?</h3>
          <p className="text-blue-100 text-sm">Upload a fundus image to get an AI diagnosis in seconds.</p>
        </div>
        <button onClick={() => onNav("scan")} className="flex items-center gap-2 bg-white text-blue-700 font-semibold px-5 py-2.5 rounded-xl hover:bg-blue-50 transition-colors text-sm flex-shrink-0">
          <Upload size={15} /> New Retina Scan
        </button>
      </div>
    </div>
  );
}

// ─── Scan Animation Card ─────────────────────────────────────────────────────
const SCAN_STEPS = [
  { min: 0, max: 15, label: "Uploading image...", icon: "upload" },
  { min: 15, max: 35, label: "Preprocessing fundus...", icon: "eye" },
  { min: 35, max: 70, label: "Running neural network...", icon: "brain" },
  { min: 70, max: 90, label: "Generating heatmap...", icon: "heatmap" },
  { min: 90, max: 100, label: "Analysis complete!", icon: "check" },
] as const;

function ScanAnimationCard({ progress, preview }: { progress: number; preview: string }) {
  const step = SCAN_STEPS.find(s => progress >= s.min && progress < s.max) ?? SCAN_STEPS[SCAN_STEPS.length - 1];
  const isComplete = progress >= 100;
  const circumference = 2 * Math.PI * 54;
  const strokeDashoffset = circumference - (progress / 100) * circumference;

  return (
    <div className="bg-card rounded-2xl border border-blue-200 shadow-md overflow-hidden" style={{ animation: "fadeInUp 0.4s ease-out" }}>
      <style>{`
        @keyframes fadeInUp {
          0% { opacity: 0; transform: translateY(12px); }
          100% { opacity: 1; transform: translateY(0); }
        }
        @keyframes scanGlow {
          0%, 100% { opacity: 0.6; }
          50% { opacity: 1; }
        }
        @keyframes completePulse {
          0% { transform: scale(0.8); opacity: 0; }
          50% { transform: scale(1.05); opacity: 1; }
          100% { transform: scale(1); opacity: 1; }
        }
      `}</style>

      {/* Header */}
      <div className="px-5 py-3 border-b border-border bg-blue-50/60 flex items-center gap-2">
        <div className={cn("w-2 h-2 rounded-full", isComplete ? "bg-emerald-500" : "bg-blue-500 animate-pulse")} />
        <span className="text-xs font-semibold text-foreground uppercase tracking-wide">
          {isComplete ? "Analysis Complete" : "AI Analysis in Progress"}
        </span>
      </div>

      <div className="p-5 flex items-center gap-6">
        {/* Left: retina thumbnail with scan line */}
        <div className="relative w-40 h-40 rounded-xl overflow-hidden flex-shrink-0 bg-slate-100">
          <img src={preview} alt="" className="w-full h-full object-cover" />
          {!isComplete && (
            <div
              className="absolute left-0 right-0 h-[2px] z-10"
              style={{
                top: `${Math.min(progress, 98)}%`,
                background: "linear-gradient(90deg, transparent 0%, #3b82f6 20%, #60a5fa 50%, #3b82f6 80%, transparent 100%)",
                boxShadow: "0 0 12px 3px rgba(59,130,246,0.4), 0 0 30px 6px rgba(59,130,246,0.15)",
                animation: "scanGlow 1.2s ease-in-out infinite",
              }}
            />
          )}
          {/* Corner brackets */}
          {[
            "top-1 left-1 border-t border-l",
            "top-1 right-1 border-t border-r",
            "bottom-1 left-1 border-b border-l",
            "bottom-1 right-1 border-b border-r",
          ].map((pos, i) => (
            <div key={i} className={`absolute ${pos} w-5 h-5 border-blue-400/70 rounded-[2px] z-10`} />
          ))}
        </div>

        {/* Right: progress ring + labels */}
        <div className="flex-1 flex flex-col items-center gap-3">
          {/* Progress ring */}
          <div className="relative w-28 h-28">
            <svg className="w-full h-full -rotate-90" viewBox="0 0 120 120">
              <circle cx="60" cy="60" r="54" fill="none" stroke="#e2e8f0" strokeWidth="5" />
              <circle
                cx="60" cy="60" r="54" fill="none"
                stroke={isComplete ? "#22c55e" : "#3b82f6"}
                strokeWidth="5"
                strokeLinecap="round"
                strokeDasharray={circumference}
                strokeDashoffset={strokeDashoffset}
                style={{ transition: "stroke-dashoffset 0.35s ease, stroke 0.3s ease" }}
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              {isComplete ? (
                <div style={{ animation: "completePulse 0.5s ease-out" }}>
                  <CheckCircle size={32} className="text-emerald-500" />
                </div>
              ) : (
                <>
                  <span className="text-2xl font-bold text-foreground tabular-nums">{progress}</span>
                  <span className="text-xs text-muted-foreground font-medium -mt-0.5">%</span>
                </>
              )}
            </div>
          </div>

          {/* Step label */}
          <div className="flex items-center gap-2 bg-blue-50 rounded-full px-3 py-1.5 border border-blue-100">
            {step.icon === "upload" && <Upload size={12} className="text-blue-500" />}
            {step.icon === "eye" && <Eye size={12} className="text-blue-500" />}
            {step.icon === "brain" && <Brain size={12} className="text-blue-500 animate-pulse" />}
            {step.icon === "heatmap" && <Activity size={12} className="text-blue-500 animate-pulse" />}
            {step.icon === "check" && <CheckCircle size={12} className="text-emerald-500" />}
            <span className="text-xs font-semibold text-foreground">{step.label}</span>
          </div>

          {/* Step indicators */}
          <div className="flex gap-1">
            {SCAN_STEPS.map((s, i) => (
              <div
                key={i}
                className="h-1 rounded-full transition-all duration-300"
                style={{
                  width: progress >= s.max ? "18px" : progress >= s.min ? "18px" : "6px",
                  backgroundColor: progress >= s.max ? "#22c55e" : progress >= s.min ? "#3b82f6" : "#e2e8f0",
                }}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Page: Retina Scan ────────────────────────────────────────────────────────
function ScanPage({ onNav, details, onDetailsChange, onScanComplete }: {
  onNav: (p: Page) => void;
  details: PatientDetails;
  onDetailsChange: (d: PatientDetails) => void;
  onScanComplete: (id: string) => void;
}) {
  const [dragging, setDragging] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [showScanAnim, setShowScanAnim] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const { scanId, scanning, progress, error: scanError, startScan } = useScan();

  const set = (field: keyof PatientDetails) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) =>
    onDetailsChange({ ...details, [field]: e.target.value });

  const todayStr = new Date().toISOString().split("T")[0];
  const nameValid = /^[A-Za-z.\s\-']+$/.test(details.fullName.trim());
  const dobValid = details.dob ? details.dob <= todayStr : false;
  const filledCount = [details.fullName.trim() && nameValid, details.dob && dobValid, details.gender, details.eye].filter(Boolean).length;
  const remainingCount = 4 - filledCount;
  const isComplete = filledCount === 4;

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (!isComplete) return;
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) { setPreview(URL.createObjectURL(file)); setImageFile(file); }
  }, [isComplete]);

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) { setPreview(URL.createObjectURL(file)); setImageFile(file); }
  };

  useEffect(() => {
    if (scanError) toast.error(scanError);
  }, [scanError]);

  const handleScan = async () => {
    if (!imageFile) { toast.error("Please select an image file first."); return; }
    setShowScanAnim(true);
    startScan({
      patient_name: toTitleCase(details.fullName || "Unknown"),
      patient_dob: details.dob || undefined,
      patient_gender: details.gender || undefined,
      patient_eye: details.eye || undefined,
      patient_mobile: details.mobile || undefined,
      patient_notes: details.notes || undefined,
      patient_id: details.patient_id || undefined,
      image: imageFile,
    }).then(id => {
      if (id) onScanComplete(id);
    });
  };

  useEffect(() => {
    if (showScanAnim && !scanning && progress >= 100) {
      const t = setTimeout(() => { setShowScanAnim(false); onNav("analysis"); }, 600);
      return () => clearTimeout(t);
    }
  }, [showScanAnim, scanning, progress, onNav]);

  return (
    <div className="max-w-5xl space-y-5">
      <div className="grid lg:grid-cols-3 gap-5">
          {/* Upload Area */}
          <div className="lg:col-span-2 space-y-4">
          {/* Gated upload drop zone */}
          <div className={cn("relative", (!isComplete || showScanAnim) && "opacity-50 pointer-events-none select-none")}>
            <div
              onDragOver={e => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              onClick={() => isComplete && fileRef.current?.click()}
              className={cn(
                "border-2 border-dashed rounded-2xl transition-all min-h-64 flex flex-col items-center justify-center relative overflow-hidden",
                isComplete ? "cursor-pointer" : "cursor-not-allowed",
                dragging && isComplete ? "border-blue-500 bg-blue-50" : "border-slate-200 hover:border-blue-400 hover:bg-blue-50/50 bg-card"
              )}
            >
              <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={handleFile} />
              {preview ? (
                <>
                  <img src={preview} alt="Retina scan preview" className="absolute inset-0 w-full h-full object-cover" />
                  <div className="absolute inset-0 bg-slate-900/30" />
                  {!showScanAnim && (
                    <div className="relative z-10 text-center">
                      <CheckCircle size={32} className="text-white mx-auto mb-2" />
                      <p className="text-white font-semibold text-sm">Image loaded — ready to scan</p>
                      <p className="text-white/70 text-xs mt-1">Click to replace</p>
                    </div>
                  )}
                </>
              ) : (
                <div className="text-center p-8">
                  <div className="w-16 h-16 rounded-2xl bg-blue-50 flex items-center justify-center mx-auto mb-4">
                    <Upload size={26} className="text-blue-500" />
                  </div>
                  <p className="font-semibold text-foreground mb-1">Drop fundus image here</p>
                  <p className="text-sm text-muted-foreground mb-4">Supports JPEG, PNG, TIFF, DICOM</p>
                  <span className="inline-flex items-center gap-2 bg-blue-600 text-white text-sm font-semibold px-4 py-2 rounded-xl">
                    <Upload size={14} /> Browse Files
                  </span>
                </div>
              )}
            </div>
            {!isComplete && (
              <div className="absolute inset-0 flex items-center justify-center rounded-2xl">
                <div className="bg-slate-800/80 backdrop-blur-sm text-white text-xs font-semibold px-4 py-2.5 rounded-xl flex items-center gap-2">
                  <Lock size={12} /> Complete {remainingCount} remaining field{remainingCount > 1 ? "s" : ""} to enable upload
                </div>
              </div>
            )}
          </div>
          {showScanAnim && preview && (
            <ScanAnimationCard progress={progress} preview={preview} />
          )}
        </div>

        {/* Patient Details — controlled form */}
        <div className="bg-card rounded-2xl border border-border p-5 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-foreground text-sm">Patient Details</h3>
            <span className={cn("text-xs font-semibold px-2 py-0.5 rounded-full border",
              isComplete ? "text-emerald-700 bg-emerald-50 border-emerald-200" : "text-amber-700 bg-amber-50 border-amber-200")}>
              {isComplete ? "✓ Ready" : `${filledCount} of 4 required fields complete`}
            </span>
          </div>

          <div>
            <label className="text-xs font-semibold text-foreground block mb-1">
              Full Name <span className="text-red-500">*</span>
            </label>
            <input
              type="text" placeholder="Patient Name" value={details.fullName} onChange={set("fullName")}
              className={cn("w-full px-3 py-2 text-sm bg-muted border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 placeholder:text-muted-foreground",
                details.fullName.trim() ? (nameValid ? "border-emerald-400" : "border-red-400") : "border-border")} />
            {details.fullName.trim() && !nameValid && (
              <p className="text-[11px] text-red-500 mt-1">Only letters, dots, hyphens, and spaces allowed</p>
            )}
          </div>

          <div>
            <label className="text-xs font-semibold text-foreground block mb-1">
              Date of Birth <span className="text-red-500">*</span>
            </label>
            <input
              type="date" max={todayStr} value={details.dob} onChange={set("dob")}
              className={cn("w-full px-3 py-2 text-sm bg-muted border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500",
                details.dob ? (dobValid ? "border-emerald-400" : "border-red-400") : "border-border")}
            />
            {details.dob && !dobValid && (
              <p className="text-[11px] text-red-500 mt-1">Date of birth cannot be in the future</p>
            )}
          </div>

          <div>
            <label className="text-xs font-semibold text-foreground block mb-1">
              Eye <span className="text-red-500">*</span>
            </label>
            <select value={details.eye} onChange={set("eye")}
              className={cn("w-full px-3 py-2 text-sm bg-muted border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500",
                details.eye ? "border-emerald-400" : "border-border")}>
              {["Left Eye (OS)", "Right Eye (OD)", "Both Eyes (OU)"].map(o => <option key={o}>{o}</option>)}
            </select>
          </div>

          <div>
            <label className="text-xs font-semibold text-foreground block mb-1">
              Gender <span className="text-red-500">*</span>
            </label>
            <select value={details.gender} onChange={set("gender")}
              className={cn("w-full px-3 py-2 text-sm bg-muted border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500",
                details.gender ? "border-emerald-400" : "border-border")}>
              <option value="">Select gender</option>
              {["Male", "Female"].map(o => <option key={o}>{o}</option>)}
            </select>
          </div>

          <div>
            <label className="text-xs font-semibold text-foreground block mb-1">Contact</label>
            <input
              type="tel" placeholder="+91 98765 43210" value={details.mobile} onChange={set("mobile")}
              className="w-full px-3 py-2 text-sm bg-muted border border-border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 placeholder:text-muted-foreground"
            />
          </div>

          <div>
            <label className="text-xs font-semibold text-foreground block mb-1">Clinical Notes</label>
            <textarea rows={3} placeholder="History of diabetes, HbA1c 8.2%..." value={details.notes} onChange={set("notes")}
              className="w-full px-3 py-2 text-sm bg-muted border border-border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none placeholder:text-muted-foreground" />
          </div>
        </div>
      </div>

      {/* Scan Button */}
      <div className={cn("bg-card rounded-2xl border border-border p-5 shadow-sm transition-opacity", showScanAnim && "opacity-50 pointer-events-none")}>
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div>
            <p className="font-semibold text-foreground text-sm">{showScanAnim ? "Analyzing..." : "Ready to analyze"}</p>
            <p className="text-xs text-muted-foreground">{showScanAnim ? `Progress: ${progress}%` : "AI analysis takes approximately 6-8 seconds"}</p>
          </div>
          <div className="flex gap-3">
            <button disabled={showScanAnim} className="border border-border text-foreground text-sm font-medium px-4 py-2.5 rounded-xl hover:bg-muted transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed">
              Save Draft
            </button>
            <button
              onClick={handleScan}
              disabled={!preview || !isComplete || scanning || showScanAnim}
              className={cn(
                "flex items-center gap-2 text-white text-sm font-semibold px-6 py-2.5 rounded-xl transition-colors",
                preview && isComplete && !scanning && !showScanAnim ? "bg-blue-600 hover:bg-blue-700 cursor-pointer" : "bg-slate-300 cursor-not-allowed"
              )}
            >
              {showScanAnim ? <><Spinner size={14} /> Analyzing…</> : scanning ? <><Spinner size={14} /> Analyzing…</> : <><Brain size={15} /> Run AI Analysis</>}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Page: AI Analysis ────────────────────────────────────────────────────────
function AnalysisPage({ onNav, details, scanId }: { onNav: (p: Page) => void; details: PatientDetails; scanId: string | null }) {
  const [activeTab, setActiveTab] = useState<"original" | "heatmap">("original");
  const { analysis, loading: analysisLoading, error: analysisError } = useAnalysis(scanId);
  const originalImg = useBlobImage(analysis?.image_url ?? null);
  const heatImg = useBlobImage(analysis?.heatmap_url ?? null);

  const hasDetails = details.patientId || details.fullName;
  const dx = analysis?.diagnosis;
  const probs = analysis?.probabilities;
  const findings = analysis?.findings;
  const recs = analysis?.recommendations;

  const primaryDisease = dx?.primary ?? null;
  const primaryDetail = dx?.detail ?? null;
  const confidence = dx?.confidence ?? null;
  const riskScore = dx?.risk_score ?? null;
  const severity = dx?.severity ?? null;
  const severityVariant = severity === "Moderate" || severity === "Mild" ? "warning" : severity === "Severe" ? "error" : "default";
  const heatmapAvailable = analysis?.heatmap_available ?? false;

  const probabilityData = probs ?? [];

  const findingData = findings ?? [];

  const recommendationData = recs ?? [];

  useEffect(() => {
    if (analysisError) toast.error("Failed to load analysis data");
  }, [analysisError]);

  if (analysisLoading && !analysis) {
    return (
      <div className="max-w-6xl space-y-5">
        <div className="bg-card rounded-2xl border border-border shadow-sm overflow-hidden">
          <Skeleton className="h-10 w-full" />
        </div>
        <div className="bg-card rounded-2xl border border-border p-5">
          <Skeleton className="h-6 w-1/3 mb-4" />
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
            {Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-4 w-full" />)}
          </div>
        </div>
        <div className="grid lg:grid-cols-5 gap-5">
          <div className="lg:col-span-2 bg-card rounded-2xl border border-border p-5">
            <Skeleton className="h-8 w-1/2 mb-4" />
            <Skeleton className="aspect-square w-full rounded-xl" />
          </div>
          <div className="lg:col-span-3 space-y-4">
            <SkeletonCard lines={4} />
            <SkeletonCard lines={5} />
            <SkeletonCard lines={3} />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl space-y-5">
      {/* Step 1 — Patient Details summary card */}
      <div className="bg-card rounded-2xl border border-border shadow-sm overflow-hidden">
        <div className="px-5 py-3 border-b border-border bg-muted/60 flex items-center gap-2">
          <div className="w-5 h-5 rounded-full bg-blue-600 flex items-center justify-center text-white text-xs font-bold">1</div>
          <span className="text-xs font-semibold text-foreground uppercase tracking-wide">Patient Details</span>
        </div>
        <div className="px-5 py-4 grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-4">
          {[
            { label: "Patient ID", value: (analysis?.patient?.patient_id ?? details.patientId) || "—" },
            { label: "Full Name", value: (analysis?.patient?.full_name ?? details.fullName) || "—" },
            { label: "Date of Birth", value: (analysis?.patient?.dob ?? details.dob) || "—" },
            { label: "Gender", value: (analysis?.patient?.gender ?? details.gender) || "—" },
            { label: "Eye", value: analysis?.patient?.eye ?? details.eye },
            { label: "Physician", value: (analysis?.patient?.physician ?? details.physician) || "—" },
          ].map(({ label, value }) => (
            <div key={label}>
              <p className="text-xs text-muted-foreground mb-0.5">{label}</p>
              <p className="text-sm font-semibold text-foreground truncate">{value}</p>
            </div>
          ))}
        </div>
        {!hasDetails && (
          <div className="px-5 pb-4">
            <p className="text-xs text-amber-600 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
              No patient data was entered on the Scan page — run a new scan to populate this summary.
            </p>
          </div>
        )}
      </div>

      {/* Step 2 — Disease Detection header */}
      <div className="bg-gradient-to-r from-blue-600 to-indigo-600 rounded-2xl p-5 text-white flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <div className="w-5 h-5 rounded-full bg-white/20 flex items-center justify-center text-white text-xs font-bold">2</div>
            <CheckCircle size={16} className="text-emerald-300" />
            <span className="text-sm font-semibold text-white/90">{analysisLoading ? "Analyzing…" : "Analysis Complete"}</span>
          </div>
          <h2 className="text-xl font-bold">
            {details.fullName ? `Patient: ${details.fullName}` : "Retinal Analysis Results"}
            {details.patientId ? ` · ${details.patientId}` : ""}
          </h2>
          <p className="text-blue-100 text-sm">{details.eye}{details.dob ? ` · DOB ${details.dob}` : ""} · Captured just now</p>
        </div>
        <div className="flex gap-2">
          <button onClick={() => onNav("reports")} className="flex items-center gap-2 bg-white text-blue-700 text-sm font-semibold px-4 py-2 rounded-xl hover:bg-blue-50 transition-colors">
            <FileText size={14} /> Generate Report
          </button>
        </div>
      </div>

      <div className="grid lg:grid-cols-5 gap-5">
        {/* Image Viewer */}
        <div className="lg:col-span-2 bg-card rounded-2xl border border-border p-5 shadow-sm">
          <div className="flex gap-2 mb-4">
            {(["original", "heatmap"] as const).map(t => (
              <button key={t} onClick={() => setActiveTab(t)} className={cn("px-3 py-1.5 rounded-lg text-xs font-semibold capitalize transition-colors",
                activeTab === t ? "bg-blue-600 text-white" : "bg-muted text-muted-foreground hover:text-foreground")}>
                {t === "heatmap" ? "Grad-CAM Heatmap" : "Original Scan"}
              </button>
            ))}
          </div>
          <div className="relative rounded-xl overflow-hidden bg-slate-900 aspect-square">
            {activeTab === "original" ? (
              originalImg.src ? (
                <img src={originalImg.src} alt="Original retinal scan" className="w-full h-full object-contain" />
              ) : (
                <div className="w-full h-full flex flex-col items-center justify-center gap-3 text-slate-400 p-6">
                  <ScanEye size={40} className="text-slate-600" />
                  <p className="text-xs text-center">No fundus image uploaded for this scan.</p>
                  {originalImg.error && <p className="text-[11px] text-red-400 text-center">{originalImg.error}</p>}
                </div>
              )
            ) : (
              heatImg.src ? (
                <img src={heatImg.src} alt="Retinal scan with Grad-CAM heatmap" className="w-full h-full object-contain" />
              ) : (
                <div className="w-full h-full flex flex-col items-center justify-center gap-3 text-slate-400 p-6">
                  <Layers size={40} className="text-slate-600" />
                  <p className="text-xs text-center">Heatmap unavailable for this scan.</p>
                  {heatImg.error && <p className="text-[11px] text-red-400 text-center">{heatImg.error}</p>}
                </div>
              )
            )}
            <div className="absolute bottom-2 left-2 right-2 bg-black/50 backdrop-blur-sm rounded-lg px-3 py-1.5 flex items-center justify-between">
              <span className="text-white text-xs">{primaryDetail ?? "—"}</span>
              <span className="text-emerald-400 text-xs font-semibold">{confidence !== null ? `${confidence}%` : "—"}</span>
            </div>
          </div>
          {activeTab === "heatmap" && (
            <div className="mt-3 flex items-center justify-between text-xs text-muted-foreground">
              <span>Low attention</span>
              <div className="flex-1 mx-3 h-2 rounded-full" style={{ background: "linear-gradient(to right, transparent, #fde68a, #f97316, #dc2626)" }} />
              <span>High attention</span>
            </div>
          )}
        </div>

          {/* Primary Result */}
          <div className="lg:col-span-3 space-y-4">
            {analysisLoading && !dx ? (
              <>
                <SkeletonCard lines={4} />
                <SkeletonCard lines={6} />
                <SkeletonCard lines={4} />
              </>
            ) : (
              <>
              <div className="bg-card rounded-2xl border border-amber-200 bg-amber-50/30 p-5 shadow-sm">
            <div className="flex items-start justify-between mb-4">
              <div>
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1">Primary Diagnosis</p>
                <h3 className="text-xl font-bold text-foreground">{primaryDisease ?? "—"}</h3>
                <p className="text-sm text-muted-foreground">{primaryDetail ?? "—"}</p>
              </div>
              <Badge variant={severityVariant}>{severity?.toUpperCase() ?? "—"}</Badge>
            </div>
            <div className="grid grid-cols-3 gap-3">
              {[
                { label: "Confidence", value: confidence !== null ? `${confidence.toFixed(1)}%` : "—", color: "text-blue-600" },
                { label: "Risk Score", value: riskScore !== null ? `${riskScore.toFixed(1)}/10` : "—", color: "text-amber-600" },
                { label: "Severity", value: severity ?? "—", color: "text-amber-600" },
              ].map(({ label, value, color }) => (
                <div key={label} className="bg-white rounded-xl border border-border p-3 text-center">
                  <p className={cn("text-lg font-bold", color)}>{value}</p>
                  <p className="text-xs text-muted-foreground mt-0.5">{label}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Disease Probabilities */}
          <div className="bg-card rounded-2xl border border-border p-5 shadow-sm">
            <h4 className="font-semibold text-foreground text-sm mb-4">Multi-Class Probabilities</h4>
            {probabilityData.length > 0 ? (
            <div className="space-y-3">
              {probabilityData.map((p: any) => (
                <div key={p.name} className="flex items-center gap-3">
                  <span className="text-xs text-muted-foreground w-44 flex-shrink-0 truncate">{p.name}</span>
                  <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                    <div className={cn("h-full rounded-full transition-all", p.color || "bg-blue-500")} style={{ width: `${p.pct}%` }} />
                  </div>
                  <span className="text-xs font-semibold text-foreground w-10 text-right">{p.pct}%</span>
                </div>
              ))}
            </div>
            ) : (
            <div className="text-center py-8 text-muted-foreground text-sm">No probability data available.</div>
            )}
          </div>

          {/* Findings */}
          <div className="bg-card rounded-2xl border border-border p-5 shadow-sm">
            <h4 className="font-semibold text-foreground text-sm mb-3">AI-Detected Findings</h4>
            {findingData.length > 0 ? (
            <div className="grid grid-cols-2 gap-2">
              {findingData.map((f: any) => (
                <div key={f.finding} className="flex items-center gap-2 p-2 rounded-lg bg-muted/50">
                  {f.severity === "success" ? <CheckCircle size={13} className="text-emerald-500 flex-shrink-0" /> :
                   f.severity === "warning" ? <AlertTriangle size={13} className="text-amber-500 flex-shrink-0" /> :
                   <XCircle size={13} className="text-red-500 flex-shrink-0" />}
                  <div>
                    <p className="text-xs font-semibold text-foreground">{f.finding}</p>
                    <p className="text-xs text-muted-foreground">{f.status}</p>
                  </div>
                </div>
              ))}
            </div>
            ) : (
            <div className="text-center py-8 text-muted-foreground text-sm">No findings available.</div>
            )}
          </div>
          </>
          )}
        </div>
      </div>

      {/* Recommendations */}
      <div className="bg-card rounded-2xl border border-border p-5 shadow-sm">
        <h4 className="font-semibold text-foreground text-sm mb-4">Clinical Recommendations</h4>
        {recommendationData.length > 0 ? (
        <div className="grid md:grid-cols-3 gap-4">
          {recommendationData.map((r: any) => {
            const Icon = ICON_MAP[r.icon] || CheckCircle;
            return (
              <div
                key={r.title}
                className="rounded-xl bg-muted/40 p-4"
              >
                <div className={cn("w-8 h-8 rounded-lg flex items-center justify-center mb-3", r.color || "text-blue-600 bg-blue-50")}>
                  <Icon size={15} />
                </div>
                <p className="font-semibold text-foreground text-sm mb-1">{r.title}</p>
                <p className="text-xs text-muted-foreground leading-relaxed">{r.desc}</p>
              </div>
            );
          })}
        </div>
        ) : (
        <div className="text-center py-8 text-muted-foreground text-sm">No recommendations available.</div>
        )}
      </div>
    </div>
  );
}

// ─── Page: Reports ────────────────────────────────────────────────────────────
const PRINT_STYLE = `
@media print {
  /* Hide all app chrome — sidebar, navbar, page header bar, action buttons */
  .no-print { display: none !important; }

  /* Hide everything outside the report card container */
  body { background: white !important; }
  @page { margin: 16mm 14mm; }

  /* Remove card cosmetics for clean paper output */
  #rs-report-card {
    box-shadow: none !important;
    border: none !important;
    border-radius: 0 !important;
    width: 100% !important;
    max-width: 100% !important;
  }

  /* Preserve dark header */
  .rs-report-header {
    background: #1e293b !important;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
    color-adjust: exact;
  }

  /* Print footer visible */
  .rs-print-footer { display: flex !important; }

  /* Keep sections together */
  .rs-report-section { page-break-inside: avoid; }
}

/* Screen: hide print footer */
.rs-print-footer { display: none; }
`;

function ReportsPage({ details, scanId }: { details: PatientDetails; scanId: string | null }) {
  const [regenerating, setRegenerating] = useState(false);
  const [printing, setPrinting] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const { report, loading: reportLoading, error: reportError } = useReport(scanId);

  const rpt = report ?? {};
  const pi = rpt.patient_info ?? {};
  const img = rpt.imaging ?? {};
  const dxRpt = rpt.diagnosis ?? {};
  const findingsRpt = rpt.findings ?? [];
  const recsRpt = rpt.recommendations ?? [];

  const originalImg = useBlobImage(img.image_url);
  const heatImg = useBlobImage(img.heatmap_url);

  const patientName = toTitleCase(pi.patient_name ?? details.fullName) || "—";
  const patientId   = (pi.patient_id ?? details.patientId) || "—";
  const age         = pi.age || "—";
  const dob         = (pi.dob ?? details.dob) || "—";
  const gender      = (pi.gender ?? details.gender) || "—";
  const eye         = (img.eye ?? details.eye) || "Left Eye (OS)";
  const physician   = img.physician || "—";
  const hospitalName = img.hospital_name || "—";
  const scanDate    = img.scan_date || rpt.generated_date || "—";
  const today       = rpt.generated_date || new Date().toLocaleDateString("en-GB", { day: "2-digit", month: "long", year: "numeric" });
  const timestamp   = rpt.generated_timestamp || new Date().toISOString();

  const reportId = rpt.report_id ?? "—";
  const severityBadge = dxRpt.diagnosis?.toLowerCase().includes("moderate") ? "warning" : dxRpt.diagnosis?.toLowerCase().includes("severe") ? "error" : "success";

  useEffect(() => {
    if (reportError) toast.error("Failed to load report data");
  }, [reportError]);

  if (reportLoading && !report) {
    return (
      <div className="max-w-4xl space-y-5">
        <div className="flex items-center justify-between no-print">
          <div className="space-y-2">
            <Skeleton className="h-5 w-48" />
            <Skeleton className="h-3 w-64" />
          </div>
          <div className="flex gap-2">
            <Skeleton className="h-9 w-28" />
            <Skeleton className="h-9 w-32" />
            <Skeleton className="h-9 w-28" />
          </div>
        </div>
        <SkeletonCard lines={8} />
      </div>
    );
  }

  const handlePrint = () => {
    setPrinting(true);
    setTimeout(() => {
      window.print();
      setPrinting(false);
    }, 280);
  };

  const handleDownloadPdf = async () => {
    setDownloading(true);
    toast.loading("Generating PDF report…", { id: "pdf" });
    try {
      const blob = await reportsApi.downloadPdf(scanId);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${reportId}.pdf`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      toast.success("PDF report downloaded.", { id: "pdf" });
    } catch (e: any) {
      toast.error(e.message || "PDF download failed.", { id: "pdf" });
    } finally {
      setDownloading(false);
    }
  };

  const handleRegenerate = () => {
    setRegenerating(true);
    toast.loading("Regenerating clinical report…", { id: "regen" });
    setTimeout(() => {
      setRegenerating(false);
      toast.success("Report regenerated successfully.", { id: "regen" });
    }, 2000);
  };

  const findingsData: any[] = findingsRpt.length > 0 ? findingsRpt : [];

  const recData: any[] = recsRpt.length > 0 ? recsRpt : [];

  const possibleCauses = dxRpt.possible_causes ?? [];
  const symptomsList = dxRpt.symptoms ?? [];
  const followUpAdvice = dxRpt.follow_up_advice ?? null;

  return (
    <div id="rs-print-root" className="max-w-4xl space-y-5">
      <style>{PRINT_STYLE}</style>
      <div className="flex items-center justify-between no-print">
        <div>
          <h2 className="text-lg font-bold text-foreground">Professional Medical Report</h2>
          <p className="text-sm text-muted-foreground">Report ID: {reportId} · Generated {timestamp}</p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={handleRegenerate}
            disabled={regenerating}
            className="flex items-center gap-2 border border-border text-sm font-medium px-4 py-2 rounded-xl hover:bg-muted transition-colors disabled:opacity-60 cursor-pointer">
            <RefreshCw size={14} className={regenerating ? "animate-spin" : ""} /> Regenerate
          </button>
          <button
            onClick={handleDownloadPdf}
            disabled={downloading}
            className="flex items-center gap-2 bg-blue-600 text-white text-sm font-semibold px-4 py-2 rounded-xl hover:bg-blue-700 transition-colors disabled:opacity-70 cursor-pointer">
            {downloading
              ? <><RefreshCw size={14} className="animate-spin" /> Generating PDF…</>
              : <><Download size={14} /> Download PDF</>}
          </button>
          <button
            onClick={handlePrint}
            disabled={printing}
            className="flex items-center gap-2 border border-blue-600 text-blue-600 text-sm font-semibold px-4 py-2 rounded-xl hover:bg-blue-50 transition-colors disabled:opacity-70 cursor-pointer">
            {printing
              ? <><RefreshCw size={14} className="animate-spin" /> Preparing report…</>
              : <><Printer size={14} /> Print Report</>}
          </button>
        </div>
      </div>

      {/* Report Card */}
      <div id="rs-report-card" className="bg-card rounded-2xl border border-border shadow-sm overflow-hidden">
        {/* Report Header */}
        <div className="rs-report-header bg-gradient-to-r from-slate-900 to-slate-800 px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-blue-600 flex items-center justify-center">
                <Eye size={18} className="text-white" />
              </div>
              <div>
                <h1 className="text-white font-bold text-lg">RetinaSense AI Clinical Suite</h1>
                <p className="text-slate-400 text-xs">{hospitalName}</p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-white text-xs font-semibold">Report ID: {reportId}</p>
              <p className="text-slate-400 text-xs">Date: {today}</p>
              <Badge variant={severityBadge}>{(dxRpt.diagnosis || "").includes("Severe") ? "Severe" : (dxRpt.diagnosis || "").includes("Mild") ? "Mild" : (dxRpt.diagnosis || "").includes("Moderate") ? "Moderate" : "—"} Severity</Badge>
            </div>
          </div>
        </div>

        <div className="px-8 py-6 space-y-6">
          {/* Patient Info & Imaging Details */}
          <div className="grid md:grid-cols-2 gap-6 pb-6 border-b border-border">
            <div>
              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">Patient Information</p>
              <dl className="space-y-2">
                {[
                  ["Patient Name", patientName],
                  ["Patient ID", patientId],
                  ["Age / Gender", `${age} / ${gender}`],
                  ["Date of Birth", dob],
                  ["Medical Record No.", pi.mrn ?? "—"],
                ].map(([k, v]) => (
                  <div key={k} className="flex gap-3">
                    <dt className="text-xs text-muted-foreground w-36 flex-shrink-0">{k}</dt>
                    <dd className="text-xs font-semibold text-foreground">{v}</dd>
                  </div>
                ))}
              </dl>
            </div>
            <div>
              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">Hospital & Imaging Details</p>
              <dl className="space-y-2">
                {[
                  ["Hospital / Center", hospitalName],
                  ["Attending Physician", physician],
                  ["Eye Examined", eye],
                  ["Scan Date", scanDate],
                  ["Camera System", img.camera ?? "—"],
                ].map(([k, v]) => (
                  <div key={k} className="flex gap-3">
                    <dt className="text-xs text-muted-foreground w-36 flex-shrink-0">{k}</dt>
                    <dd className="text-xs font-semibold text-foreground">{v}</dd>
                  </div>
                ))}
              </dl>
            </div>
          </div>

          {/* Fundus Images & Grad-CAM Heatmap Section */}
          <div className="pb-6 border-b border-border">
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">Retinal Imaging & Grad-CAM Attention</p>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-slate-900 rounded-xl p-3 text-center">
                <p className="text-xs text-slate-300 font-semibold mb-2">Original Fundus Image</p>
                <div className="aspect-square rounded-lg overflow-hidden bg-black flex items-center justify-center">
                  {originalImg.src ? (
                    <img src={originalImg.src} alt="Original fundus scan" className="w-full h-full object-contain" />
                  ) : (
                    <span className="text-xs text-slate-500">Image loaded from secure store</span>
                  )}
                </div>
              </div>
              <div className="bg-slate-900 rounded-xl p-3 text-center">
                <p className="text-xs text-slate-300 font-semibold mb-2">Grad-CAM Attention Heatmap</p>
                <div className="aspect-square rounded-lg overflow-hidden bg-black flex items-center justify-center">
                  {heatImg.src ? (
                    <img src={heatImg.src} alt="Grad-CAM Heatmap" className="w-full h-full object-contain" />
                  ) : (
                    <span className="text-xs text-slate-500">Heatmap generated by neural attention</span>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Diagnosis */}
          <div className="pb-6 border-b border-border">
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">AI Diagnosis & Clinical Summary</p>
            <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 space-y-3">
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="font-bold text-amber-900 text-base">{dxRpt.diagnosis || "—"}</h3>
                  <p className="text-xs text-amber-700 mt-0.5">{dxRpt.icd10 ? `ICD-10: ${dxRpt.icd10}` : ""} {dxRpt.etdrs_grade ? ` · ETDRS Grade: ${dxRpt.etdrs_grade}` : ""}</p>
                </div>
                <div className="text-right">
                  <p className="text-lg font-bold text-amber-700">{dxRpt.confidence ? `${(dxRpt.confidence * 100 > 1 ? dxRpt.confidence : dxRpt.confidence * 100).toFixed(1)}%` : "—"}</p>
                  <p className="text-xs text-amber-600">AI Confidence</p>
                </div>
              </div>
              <p className="text-xs text-amber-800 leading-relaxed">{dxRpt.summary || "—"}</p>
            </div>
          </div>

          {/* Possible Causes & Symptoms */}
          <div className="grid md:grid-cols-2 gap-6 pb-6 border-b border-border">
            <div>
              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">Possible Clinical Causes</p>
              {possibleCauses.length > 0 ? (
              <ul className="list-disc pl-4 space-y-1 text-xs text-foreground">
                {possibleCauses.map((cause: string, i: number) => (
                  <li key={i}>{cause}</li>
                ))}
              </ul>
              ) : (
              <p className="text-xs text-muted-foreground">No causes reported.</p>
              )}
            </div>
            <div>
              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">Associated Symptoms</p>
              {symptomsList.length > 0 ? (
              <ul className="list-disc pl-4 space-y-1 text-xs text-foreground">
                {symptomsList.map((sym: string, i: number) => (
                  <li key={i}>{sym}</li>
                ))}
              </ul>
              ) : (
              <p className="text-xs text-muted-foreground">No symptoms reported.</p>
              )}
            </div>
          </div>

          {/* Findings Table */}
          <div className="pb-6 border-b border-border">
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">Detailed Biomarker Findings</p>
            {findingsData.length > 0 ? (
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-muted/50">
                  <th className="text-left px-3 py-2 rounded-tl-lg font-semibold text-muted-foreground">Finding</th>
                  <th className="text-left px-3 py-2 font-semibold text-muted-foreground">Status</th>
                  <th className="text-left px-3 py-2 font-semibold text-muted-foreground">Confidence</th>
                  <th className="text-left px-3 py-2 rounded-tr-lg font-semibold text-muted-foreground">Clinical Significance</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {findingsData.map((f: any) => (
                  <tr key={f.finding} className="hover:bg-muted/30">
                    <td className="px-3 py-2 font-medium text-foreground">{f.finding}</td>
                    <td className="px-3 py-2">
                      <Badge variant={f.status === "Absent" || f.status === "Not detected" ? "success" : f.status === "Suspected" ? "error" : "warning"}>{f.status}</Badge>
                    </td>
                    <td className="px-3 py-2 font-mono text-foreground">{f.confidence}</td>
                    <td className="px-3 py-2 text-muted-foreground">{f.significance}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            ) : (
            <p className="text-xs text-muted-foreground text-center py-4">No findings available.</p>
            )}
          </div>

          {/* Recommendations & Follow-up Advice */}
          <div className="pb-6 border-b border-border space-y-4">
            <div>
              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">Clinical Recommendations</p>
              {recData.length > 0 ? (
              <div className="space-y-2">
                {recData.map((r: any) => (
                  <div key={r.priority || r.text} className="flex items-start gap-3 p-3 rounded-xl bg-muted/40">
                    <Badge variant={r.priority === "URGENT" ? "error" : r.priority === "HIGH" ? "warning" : r.priority === "MEDIUM" ? "info" : "default"}>{r.priority}</Badge>
                    <p className="text-xs text-foreground leading-relaxed">{r.text}</p>
                  </div>
                ))}
              </div>
              ) : (
              <p className="text-xs text-muted-foreground text-center py-4">No recommendations available.</p>
              )}
            </div>
            <div className="p-3 rounded-xl bg-blue-50 border border-blue-200">
              <p className="text-xs font-bold text-blue-900 mb-1">Follow-up Advice</p>
              <p className="text-xs text-blue-800">{followUpAdvice ?? "—"}</p>
            </div>
          </div>

          {/* Medical Disclaimer */}
          <div className="rs-report-section text-xs text-muted-foreground space-y-1">
            <p><strong className="text-foreground">Medical Disclaimer:</strong> This report is generated by RetinaSense AI deep learning software and is intended to assist, not replace, clinical judgment. All findings must be reviewed and interpreted by a qualified ophthalmologist before clinical decisions are made.</p>
            <p>RetinaSense AI v3.2.1 · Model: EfficientNetB3 Retinal Classifier · Sensitivity 96.8% / Specificity 97.1%</p>
          </div>

          {/* Print-only footer */}
          <div className="rs-print-footer mt-6 pt-4 border-t border-slate-200 text-xs text-slate-500 flex items-center justify-between">
            <span>Generated by RetinaSense AI · Report ID: {reportId} · Center: {hospitalName}</span>
            <span>Generated at: {timestamp}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Page: Patients ───────────────────────────────────────────────────────────
function PatientsPage({ onNav, onViewReport, onNewScan }: { onNav: (p: Page) => void; onViewReport: (scanId: string) => void; onNewScan: (patient: Patient) => void }) {
  const { patients: patientList, create: createPatient, remove: deletePatient, loading: patientsLoading, error: patientsError } = usePatients();
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("All");
  const [showAdd, setShowAdd] = useState(false);
  const [showFilter, setShowFilter] = useState(false);
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
  const [advFilters, setAdvFilters] = useState({ diagnosis: "", risk: "", dateFrom: "", dateTo: "" });
  const [deleteTarget, setDeleteTarget] = useState<Patient | null>(null);
  const [moreMenuId, setMoreMenuId] = useState<string | null>(null);

  const confirmDelete = (p: Patient) => { setDeleteTarget(p); setMoreMenuId(null); };
  const doDelete = () => {
    if (!deleteTarget) return;
    const removed = deleteTarget;
    deletePatient(removed.id).catch(() => { toast.error(`Failed to delete ${removed.name}. Please try again.`); });
    setDeleteTarget(null);
    const toastId = toast.success(`${removed.name} has been removed.`, {
      action: { label: "Undo", onClick: () => { toast.dismiss(toastId); } },
      duration: 6000,
    });
  };

  useEffect(() => { if (patientsError) toast.error(patientsError); }, [patientsError]);

  const filtered = patientList.filter(p => {
    const matchSearch = p.name.toLowerCase().includes(search.toLowerCase()) || p.id.toLowerCase().includes(search.toLowerCase());
    const matchStatus = statusFilter === "All" || p.status === statusFilter;
    const matchDiag = !advFilters.diagnosis || p.condition === advFilters.diagnosis;
    const matchRisk = !advFilters.risk || p.risk === advFilters.risk;
    const matchFrom = !advFilters.dateFrom || p.lastScan >= advFilters.dateFrom;
    const matchTo = !advFilters.dateTo || p.lastScan <= advFilters.dateTo;
    return matchSearch && matchStatus && matchDiag && matchRisk && matchFrom && matchTo;
  });

  const statusColor = (s: string) => s === "Critical" ? "error" : s === "Active" ? "info" : "success";
  const hasAdvFilter = advFilters.diagnosis || advFilters.risk || advFilters.dateFrom || advFilters.dateTo;

  return (
    <>
      <AddPatientModal open={showAdd} onClose={() => setShowAdd(false)} onAdd={createPatient} />
      <PatientDetailDrawer patient={selectedPatient} open={!!selectedPatient} onClose={() => setSelectedPatient(null)} onNav={onNav} onViewReport={onViewReport} onNewScan={onNewScan} />

      {/* Delete confirmation modal */}
      {deleteTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <div className="absolute inset-0 bg-black/50" onClick={() => setDeleteTarget(null)} />
          <div className="relative bg-card rounded-2xl shadow-2xl w-full max-w-md p-6 z-10">
            <div className="w-10 h-10 rounded-xl bg-red-50 flex items-center justify-center mb-4">
              <Trash2 size={18} className="text-red-600" />
            </div>
            <h2 className="text-base font-bold text-foreground mb-2">Delete patient record?</h2>
            <p className="text-sm text-muted-foreground mb-6 leading-relaxed">
              This will permanently remove <strong className="text-foreground">{toTitleCase(deleteTarget.name)}</strong> (ID: {deleteTarget.id}) and all associated scans and reports. This action cannot be undone.
            </p>
            <div className="flex gap-3 justify-end">
              <button onClick={() => setDeleteTarget(null)} className="px-4 py-2 text-sm font-medium border border-border rounded-xl hover:bg-muted transition-colors cursor-pointer">
                Cancel
              </button>
              <button onClick={doDelete} className="px-4 py-2 text-sm font-semibold bg-red-600 text-white rounded-xl hover:bg-red-700 transition-colors cursor-pointer">
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Click-outside to close more-menu */}
      {moreMenuId && <div className="fixed inset-0 z-10" onClick={() => setMoreMenuId(null)} />}

    <div className="space-y-5">
      {/* Stats Row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { label: "Total Patients", value: patientList.length.toLocaleString(), icon: Users, color: "bg-blue-600" },
          { label: "Active This Month", value: patientList.filter(p => p.status === "Active").length.toString(), icon: Activity, color: "bg-indigo-600" },
          { label: "Critical Cases", value: patientList.filter(p => p.status === "Critical").length.toString(), icon: AlertTriangle, color: "bg-red-500" },
          { label: "High Risk", value: patientList.filter(p => p.risk === "High" || parseFloat(p.risk) >= 7.0).length.toString(), icon: Target, color: "bg-amber-500" },
        ].map(({ label, value, icon: Icon, color }) => (
          <div
            key={label}
            className="bg-card rounded-2xl border border-border p-4 shadow-sm flex items-center gap-3"
          >
            <div className={cn("w-9 h-9 rounded-xl flex items-center justify-center", color)}>
              <Icon size={16} className="text-white" />
            </div>
            <div>
              <p className="text-lg font-bold text-foreground">{value}</p>
              <p className="text-xs text-muted-foreground">{label}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Table */}
      <div className="bg-card rounded-2xl border border-border shadow-sm">
        <div className="flex items-center justify-between p-4 border-b border-border flex-wrap gap-3">
          <div className="relative">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
            <input
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search patients..."
              className="pl-9 pr-4 py-2 text-sm bg-muted border border-transparent rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 w-56"
            />
          </div>
          <div className="flex items-center gap-2">
            <div className="flex bg-muted rounded-xl p-0.5 text-xs font-semibold">
              {["All", "Active", "Critical", "Stable"].map(s => (
                <button
                  key={s}
                  onClick={() => setStatusFilter(s)}
                  className={cn("px-3 py-1.5 rounded-lg transition-colors", statusFilter === s ? "bg-white text-foreground shadow-sm" : "text-muted-foreground")}
                >
                  {s}
                </button>
              ))}
            </div>
            <div className="relative">
              <button
                onClick={() => setShowFilter(v => !v)}
                className={cn(
                  "flex items-center gap-1.5 border text-sm font-medium px-3 py-2 rounded-xl hover:bg-muted transition-colors",
                  hasAdvFilter ? "border-blue-400 text-blue-700 bg-blue-50" : "border-border"
                )}
              >
                <Filter size={13} /> Filter {hasAdvFilter && <span className="w-1.5 h-1.5 rounded-full bg-blue-600 ml-0.5" />}
              </button>
              <FilterPopover open={showFilter} onClose={() => setShowFilter(false)} filters={advFilters} onChange={setAdvFilters} />
            </div>
            <button
              onClick={() => setShowAdd(true)}
              className="flex items-center gap-1.5 bg-blue-600 text-white text-sm font-semibold px-3 py-2 rounded-xl hover:bg-blue-700 transition-colors">
              <Plus size={13} /> Add Patient
            </button>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="bg-muted/40">
                {["Patient", "Age", "Condition", "Severity", "Last Scan", "Risk", "Status", ""].map(h => (
                  <th key={h} className="text-left px-4 py-3 text-xs font-semibold text-muted-foreground first:pl-5">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {filtered.map(p => (
                <tr
                  key={p.id}
                  className="hover:bg-muted/30 transition-colors cursor-pointer"
                  onClick={() => setSelectedPatient(p)}
                >
                  <td className="px-5 py-3">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-400 to-indigo-500 flex items-center justify-center flex-shrink-0">
                        <span className="text-white text-xs font-bold">{toTitleCase(p.name).split(" ").map(n => n[0]).join("")}</span>
                      </div>
                      <div>
                        <p className="text-sm font-semibold text-foreground">{toTitleCase(p.name)}</p>
                        <p className="text-xs text-muted-foreground font-mono">{p.id}</p>
                      </div>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-sm text-foreground">{p.age > 0 && p.age <= 120 ? p.age : "—"}</td>
                  <td className="px-4 py-3 text-sm text-foreground">{p.condition}</td>
                  <td className="px-4 py-3"><Badge variant={p.severity === "Severe" ? "error" : p.severity === "Moderate" ? "warning" : p.severity === "None" ? "success" : "info"}>{p.severity}</Badge></td>
                  <td className="px-4 py-3 text-xs text-muted-foreground font-mono">{p.lastScan}</td>
                  <td className="px-4 py-3"><Badge variant={riskColor(p.risk) as any}>{p.risk}</Badge></td>
                  <td className="px-4 py-3"><Badge variant={statusColor(p.status) as any}>{p.status}</Badge></td>
                  <td className="px-4 py-3" onClick={e => e.stopPropagation()}>
                    <div className="flex items-center gap-1">
                      <button
                        onClick={() => {
                          setScanDetails({
                            fullName: p.name, dob: p.date_of_birth || "", eye: "Left Eye (OS)",
                            gender: p.gender || "", mobile: p.mobile || "", notes: p.notes || "", patient_id: p.id,
                          });
                          onNav("scan");
                        }}
                        className="p-1.5 rounded-lg hover:bg-muted transition-colors text-muted-foreground hover:text-foreground" title="New scan">
                        <ScanEye size={14} />
                      </button>
                      <button
                        onClick={() => { setSelectedPatient(p); }}
                        className="p-1.5 rounded-lg hover:bg-muted transition-colors text-muted-foreground hover:text-foreground" title="View reports">
                        <FileText size={14} />
                      </button>
                      <button
                        onClick={async () => {
                          toast.loading("Generating PDF…", { id: "pdf" });
                          try {
                            const blob = await patientsApi.downloadReportPdf(p.id);
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement("a");
                            a.href = url;
                            a.download = `${p.name.replace(/\s+/g, "_")}_report.pdf`;
                            document.body.appendChild(a);
                            a.click();
                            document.body.removeChild(a);
                            URL.revokeObjectURL(url);
                            toast.success(`Report downloaded for ${p.name}.`, { id: "pdf" });
                          } catch (e: any) {
                            toast.error(e.message || "PDF download failed.", { id: "pdf" });
                          }
                        }}
                        className="p-1.5 rounded-lg hover:bg-muted transition-colors text-muted-foreground hover:text-foreground" title="Download report">
                        <Download size={14} />
                      </button>
                      <button
                        onClick={() => confirmDelete(p)}
                        className="p-1.5 rounded-lg hover:bg-red-50 hover:text-red-600 transition-colors text-muted-foreground" title="Delete patient">
                        <Trash2 size={14} />
                      </button>
                      {/* More menu */}
                      <div className="relative">
                        <button
                          onClick={() => setMoreMenuId(moreMenuId === p.id ? null : p.id)}
                          className="p-1.5 rounded-lg hover:bg-muted transition-colors text-muted-foreground hover:text-foreground" title="More options">
                          <MoreHorizontal size={14} />
                        </button>
                        {moreMenuId === p.id && (
                          <div className="absolute right-0 top-8 z-20 bg-white border border-border rounded-xl shadow-xl w-44 py-1 text-sm">
                            <button
                              onClick={() => { setMoreMenuId(null); setSelectedPatient(p); }}
                              className="w-full text-left px-3 py-2 hover:bg-muted transition-colors flex items-center gap-2">
                              <User size={13} className="text-muted-foreground" /> View profile
                            </button>
                            <button
                              onClick={async () => {
                                setMoreMenuId(null);
                                toast.loading("Generating PDF…", { id: "pdf" });
                                try {
                                  const blob = await patientsApi.downloadReportPdf(p.id);
                                  const url = URL.createObjectURL(blob);
                                  const a = document.createElement("a");
                                  a.href = url;
                                  a.download = `${p.name.replace(/\s+/g, "_")}_report.pdf`;
                                  document.body.appendChild(a);
                                  a.click();
                                  document.body.removeChild(a);
                                  URL.revokeObjectURL(url);
                                  toast.success(`Report downloaded for ${p.name}.`, { id: "pdf" });
                                } catch (e: any) {
                                  toast.error(e.message || "PDF download failed.", { id: "pdf" });
                                }
                              }}
                              className="w-full text-left px-3 py-2 hover:bg-muted transition-colors flex items-center gap-2">
                              <Download size={13} className="text-muted-foreground" /> Export record
                            </button>
                            <div className="border-t border-border my-1" />
                            <button
                              onClick={() => confirmDelete(p)}
                              className="w-full text-left px-3 py-2 hover:bg-red-50 text-red-600 transition-colors flex items-center gap-2">
                              <Trash2 size={13} /> Delete patient
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                  </td>
                </tr>
              ))}
              {filtered.length === 0 && (
                <tr>
                  <td colSpan={8} className="px-5 py-10 text-center text-sm text-muted-foreground">
                    No patients match your current filters.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        <div className="px-5 py-3 border-t border-border flex items-center justify-between text-xs text-muted-foreground">
          <span>Showing {filtered.length} of {patientList.length} patients</span>
          <div className="flex items-center gap-1">
            {[1, 2, 3, "...", 41].map((p, i) => (
              <button key={i} className={cn("w-7 h-7 rounded-lg text-xs font-medium transition-colors",
                p === 1 ? "bg-blue-600 text-white" : "hover:bg-muted text-muted-foreground")}>{p}</button>
            ))}
          </div>
        </div>
      </div>
    </div>
    </>
  );
}

// ─── Page: Settings ───────────────────────────────────────────────────────────
function SettingsPage() {
  const [activeTab, setActiveTab] = useState("profile");
  const [darkMode, setDarkMode] = useState(false);
  const [notifications, setNotifications] = useState({
    criticalAlerts: true,
    reportReady: true,
    weeklyDigest: false,
    modelUpdates: true,
  });
  const [profileForm, setProfileForm] = useState({
    first_name: "", last_name: "", email: "", phone: "", specialty: "", institution: "", hospital_name: "",
  });
  const [passwordForm, setPasswordForm] = useState({ current_password: "", new_password: "", confirm_password: "" });
  const [language, setLanguage] = useState("English (US)");
  const [reportLanguage, setReportLanguage] = useState("English — Clinical Standard");
  const [avatarPreview, setAvatarPreview] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const settings = useSettings();

  useEffect(() => {
    if (settings.notifications) {
      setNotifications((prev) => ({ ...prev, ...settings.notifications }));
    }
  }, [settings.notifications]);

  useEffect(() => {
    if (settings.profile?.dark_mode !== undefined) {
      setDarkMode(settings.profile.dark_mode);
    }
  }, [settings.profile]);

  useEffect(() => {
    if (settings.profile) {
      setProfileForm({
        first_name: settings.profile.first_name ?? "",
        last_name: settings.profile.last_name ?? "",
        email: settings.profile.email ?? "",
        phone: settings.profile.phone ?? "",
        specialty: settings.profile.specialty ?? "",
        institution: settings.profile.institution ?? "",
        hospital_name: settings.profile.hospital_name ?? "",
      });
    }
  }, [settings.profile]);

  useEffect(() => {
    if (settings.error) toast.error("Failed to load settings");
  }, [settings.error]);

  const setProfile = (k: string, v: string) => setProfileForm(f => ({ ...f, [k]: v }));

  const tabs = [
    { id: "profile", label: "Profile", icon: User },
    { id: "security", label: "Security", icon: Lock },
    { id: "notifications", label: "Notifications", icon: Bell },
    { id: "appearance", label: "Appearance", icon: Sun },
  ];

  return (
    <div className="max-w-4xl">
      {/* Tab bar (horizontal scroll on mobile, sidebar on desktop) */}
      <div className="flex overflow-x-auto md:hidden gap-1 mb-4 pb-1">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            className={cn(
              "flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium whitespace-nowrap transition-all flex-shrink-0",
              activeTab === id ? "bg-blue-50 text-blue-700" : "text-muted-foreground hover:bg-muted hover:text-foreground"
            )}
          >
            <Icon size={15} />
            {label}
          </button>
        ))}
      </div>

      <div className="flex gap-5">
        {/* Tab Sidebar (desktop only) */}
        <div className="hidden md:block w-44 flex-shrink-0">
          <nav className="space-y-0.5">
            {tabs.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id)}
                className={cn(
                  "w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all",
                  activeTab === id ? "bg-blue-50 text-blue-700" : "text-muted-foreground hover:bg-muted hover:text-foreground"
                )}
              >
                <Icon size={15} />
                {label}
              </button>
            ))}
          </nav>
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0 space-y-4">
          {activeTab === "profile" && (
            <>
              <div className="bg-card rounded-2xl border border-border p-6 shadow-sm">
                <h3 className="font-semibold text-foreground mb-5">Profile Information</h3>
                <div className="flex items-start gap-5 mb-6 pb-6 border-b border-border">
                  <button onClick={() => fileInputRef.current?.click()} className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center flex-shrink-0 overflow-hidden hover:ring-2 hover:ring-blue-400 transition-all cursor-pointer" title="Click to change photo">
                    {avatarPreview ? (
                      <img src={avatarPreview} alt="Avatar" className="w-full h-full object-cover" />
                    ) : (
                      <span className="text-white text-xl font-bold">{(settings.profile?.first_name?.[0] ?? "") + (settings.profile?.last_name?.[0] ?? "") || "—"}</span>
                    )}
                  </button>
                  <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={(e) => { const file = e.target.files?.[0]; if (file) { const reader = new FileReader(); reader.onload = (ev) => { setAvatarPreview(ev.target?.result as string); toast.success("Photo updated."); }; reader.readAsDataURL(file); } }} />
                  <div>
                    <p className="font-bold text-foreground">{settings.profile?.first_name || settings.profile?.last_name ? `Dr. ${settings.profile?.first_name ?? ""} ${settings.profile?.last_name ?? ""}` : "—"}</p>
                    <p className="text-sm text-muted-foreground">{settings.profile?.specialty || "—"}{settings.profile?.institution ? ` · ${settings.profile.institution}` : ""}</p>
                    <button onClick={() => fileInputRef.current?.click()} className="mt-2 text-xs text-blue-600 font-semibold hover:text-blue-700">Change photo</button>
                  </div>
                </div>
                <div className="grid md:grid-cols-2 gap-4">
                  {[
                    { label: "First Name", key: "first_name", icon: User },
                    { label: "Last Name", key: "last_name", icon: User },
                    { label: "Email", key: "email", icon: Mail },
                    { label: "Phone", key: "phone", icon: Phone },
                    { label: "Specialty", key: "specialty", icon: Microscope },
                    { label: "Institution", key: "institution", icon: Building },
                  ].map(({ label, key, icon: Icon }) => (
                    <div key={key}>
                      <label className="text-xs font-semibold text-foreground block mb-1.5">{label}</label>
                      <div className="relative">
                        <Icon size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
                        <input value={profileForm[key as keyof typeof profileForm]} onChange={e => setProfile(key, e.target.value)} className="w-full pl-9 pr-4 py-2 text-sm bg-muted border border-border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500" />
                      </div>
                    </div>
                  ))}
                </div>
                <div className="mt-4">
                  <label className="text-xs font-semibold text-foreground block mb-1.5">Hospital / Center Name</label>
                  <div className="relative">
                    <Building size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
                    <input value={profileForm.hospital_name} onChange={e => setProfile("hospital_name", e.target.value)} placeholder="Your Hospital & RetinaSense Clinical Center" className="w-full pl-9 pr-4 py-2 text-sm bg-muted border border-border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500" />
                  </div>
                  <p className="text-[11px] text-muted-foreground mt-1">Shown on all generated reports</p>
                </div>
                <div className="mt-4 flex gap-2">
                  <button onClick={async () => { try { await settings.updateProfile(profileForm); await settings.refetchProfile(); toast.success("Profile saved successfully."); } catch { toast.error("Failed to save profile."); } }} className="bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold px-4 py-2 rounded-xl transition-colors">Save Changes</button>
                  <button onClick={() => { if (settings.profile) { setProfileForm({ first_name: settings.profile.first_name ?? "", last_name: settings.profile.last_name ?? "", email: settings.profile.email ?? "", phone: settings.profile.phone ?? "", specialty: settings.profile.specialty ?? "", institution: settings.profile.institution ?? "", hospital_name: settings.profile.hospital_name ?? "" }); toast.info("Changes discarded."); } }} className="border border-border text-sm font-medium px-4 py-2 rounded-xl hover:bg-muted transition-colors">Cancel</button>
                </div>
              </div>
            </>
          )}

          {activeTab === "security" && (
            <div className="bg-card rounded-2xl border border-border p-6 shadow-sm">
              <h3 className="font-semibold text-foreground mb-5">Security Settings</h3>
              <div className="space-y-4">
                <div className="pb-4 border-b border-border">
                  <h4 className="text-sm font-semibold text-foreground mb-3">Change Password</h4>
                  {([
                    { label: "Current Password", key: "current_password" },
                    { label: "New Password", key: "new_password" },
                    { label: "Confirm New Password", key: "confirm_password" },
                  ] as const).map(({ label, key }) => (
                    <div key={label} className="mb-3">
                      <label className="text-xs font-semibold text-foreground block mb-1.5">{label}</label>
                      <div className="relative">
                        <Lock size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
                        <input
                          type="password"
                          placeholder="••••••••"
                          value={passwordForm[key]}
                          onChange={(e) => setPasswordForm(p => ({ ...p, [key]: e.target.value }))}
                          className="w-full pl-9 pr-4 py-2 text-sm bg-muted border border-border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                      </div>
                    </div>
                  ))}
                  <button onClick={async () => {
                    if (passwordForm.new_password !== passwordForm.confirm_password) { toast.error("Passwords do not match."); return; }
                    if (passwordForm.new_password.length < 8) { toast.error("Password must be at least 8 characters."); return; }
                    try {
                      await settings.changePassword(passwordForm.current_password, passwordForm.new_password);
                      toast.success("Password updated successfully.");
                      setPasswordForm({ current_password: "", new_password: "", confirm_password: "" });
                    } catch { toast.error("Failed to update password."); }
                  }} className="bg-blue-600 text-white text-sm font-semibold px-4 py-2 rounded-xl hover:bg-blue-700 transition-colors mt-1">Update Password</button>
                </div>
                <div className="pb-4 border-b border-border">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-semibold text-foreground">Two-Factor Authentication</p>
                      <p className="text-xs text-muted-foreground">Add an extra layer of security to your account</p>
                    </div>
                    <Badge variant="success">Enabled</Badge>
                  </div>
                </div>
                <div>
                  <p className="text-sm font-semibold text-foreground mb-3">Active Sessions</p>
                  {[
                    { device: "MacBook Pro 14\"", location: "Mumbai, IN", time: "Current session", active: true },
                    { device: "iPhone 15 Pro", location: "Mumbai, IN", time: "2 hours ago", active: false },
                    { device: "iPad Pro", location: "Delhi, IN", time: "3 days ago", active: false },
                  ].map(({ device, location, time, active }) => (
                    <div key={device} className="flex items-center justify-between py-3 border-b border-border last:border-0">
                      <div>
                        <p className="text-sm font-medium text-foreground">{device}</p>
                        <p className="text-xs text-muted-foreground">{location} · {time}</p>
                      </div>
                      {active ? <Badge variant="success">Active</Badge> : <button onClick={() => toast.success(`Session on ${device} revoked.`)} className="text-xs text-red-500 font-medium hover:text-red-600">Revoke</button>}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {activeTab === "notifications" && (
            <div className="bg-card rounded-2xl border border-border p-6 shadow-sm">
              <h3 className="font-semibold text-foreground mb-5">Notification Preferences</h3>
              <div className="space-y-4">
                {[
                  { key: "criticalAlerts", title: "Critical Patient Alerts", desc: "Immediate notification for high-severity AI detections requiring urgent action" },
                  { key: "reportReady", title: "Report Ready", desc: "Notify when AI analysis report is generated and ready for review" },
                  { key: "weeklyDigest", title: "Weekly Clinical Digest", desc: "Summary of scan activity, disease trends, and model performance" },
                  { key: "modelUpdates", title: "Model Updates", desc: "Notifications when AI models are updated with improved performance" },
                ].map(({ key, title, desc }) => (
                  <div key={key} className="flex items-start justify-between py-3 border-b border-border last:border-0">
                    <div>
                      <p className="text-sm font-semibold text-foreground">{title}</p>
                      <p className="text-xs text-muted-foreground leading-relaxed max-w-sm">{desc}</p>
                    </div>
                    <button
                      onClick={() => {
                        const next = !notifications[key as keyof typeof notifications];
                        setNotifications(n => ({ ...n, [key]: next }));
                        settings.updateNotifications({ ...notifications, [key]: next }).catch(() => { toast.error("Failed to update notification settings."); });
                      }}
                      className={cn(
                        "w-10 h-5.5 rounded-full transition-colors flex-shrink-0 relative mt-0.5",
                        notifications[key as keyof typeof notifications] ? "bg-blue-600" : "bg-muted"
                      )}
                      style={{ height: "22px", minWidth: "40px" }}
                    >
                      <span className={cn(
                        "w-4 h-4 bg-white rounded-full absolute top-[3px] transition-all shadow-sm",
                        notifications[key as keyof typeof notifications] ? "left-[22px]" : "left-[3px]"
                      )} />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === "appearance" && (
            <div className="bg-card rounded-2xl border border-border p-6 shadow-sm">
              <h3 className="font-semibold text-foreground mb-5">Appearance</h3>
              <div className="space-y-5">
                <div>
                  <p className="text-sm font-semibold text-foreground mb-3">Theme</p>
                  <div className="grid grid-cols-2 gap-3">
                    {[
                      { id: "light", label: "Light", icon: Sun, preview: "bg-white border-slate-200" },
                      { id: "dark", label: "Dark", icon: Moon, preview: "bg-slate-900 border-slate-700" },
                    ].map(({ id, label, icon: Icon, preview }) => (
                      <button
                        key={id}
                        onClick={async () => { setDarkMode(id === "dark"); applyTheme(id === "dark"); try { await settings.toggleTheme(id === "dark"); } catch { toast.error("Failed to update theme."); } toast.success(`${label} theme selected.`); }}
                        className={cn(
                          "border-2 rounded-xl p-4 text-left transition-all",
                          (id === "dark") === darkMode ? "border-blue-500 bg-blue-50" : "border-border hover:border-slate-300"
                        )}
                      >
                        <div className={cn("w-full h-16 rounded-lg mb-3 border", preview)} />
                        <div className="flex items-center gap-2">
                          <Icon size={14} className="text-muted-foreground" />
                          <span className="text-sm font-medium text-foreground">{label}</span>
                          {(id === "dark") === darkMode && <CheckCircle size={13} className="text-blue-600 ml-auto" />}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
                <div className="pb-4 border-b border-border">
                  <p className="text-sm font-semibold text-foreground mb-1">Language & Region</p>
                  <select value={language} onChange={(e) => setLanguage(e.target.value)} className="mt-2 w-full px-3 py-2 text-sm bg-muted border border-border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500">
                    <option>English (US)</option>
                    <option>English (UK)</option>
                    <option>Hindi</option>
                    <option>German</option>
                  </select>
                </div>
                <div>
                  <p className="text-sm font-semibold text-foreground mb-1">AI Report Language</p>
                  <select value={reportLanguage} onChange={(e) => setReportLanguage(e.target.value)} className="mt-2 w-full px-3 py-2 text-sm bg-muted border border-border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500">
                    <option>English — Clinical Standard</option>
                    <option>English — Patient-Friendly</option>
                    <option>Hindi</option>
                  </select>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Root App ─────────────────────────────────────────────────────────────────

export default function App() {
  const { user, isAuthenticated, login: authLogin, logout: authLogout } = useAuth();
  const [page, setPage] = useState<Page>("dashboard");
  const [scanId, setScanId] = useState<string | null>(null);
  const [scanDetails, setScanDetails] = useState<PatientDetails>(EMPTY_PATIENT_DETAILS);
  const settings = useSettings();

  useEffect(() => {
    applyTheme(settings.profile?.dark_mode === true);
  }, [settings.profile?.dark_mode]);

  useEffect(() => {
    const onExpired = () => {
      authLogout();
      setPage("login");
    };
    window.addEventListener(AUTH_EXPIRED_EVENT, onExpired);
    return () => window.removeEventListener(AUTH_EXPIRED_EVENT, onExpired);
  }, [authLogout]);

  const handleLogin = async (email: string, password: string) => {
    const ok = await authLogin(email, password);
    if (ok) setPage("dashboard");
    return ok;
  };
  const handleLogout = () => { authLogout(); setPage("dashboard"); };

  const defaultUser: AuthUser = user ?? { name: "", role: "", initials: "", email: "" };

  if (page === "login") return <><Toaster position="top-right" richColors /><LoginPage onNav={setPage} onLogin={handleLogin} /></>;
  if (page === "signup") return <><Toaster position="top-right" richColors /><SignUpPage onNav={setPage} /></>;

  return (
    <>
      <Toaster position="top-right" richColors />
      <AppShell
        page={page}
        onNav={setPage}
        isLoggedIn={isAuthenticated}
        user={defaultUser}
        onLogin={() => setPage("login")}
        onLogout={handleLogout}
      >
        {page === "dashboard" && <DashboardPage onNav={setPage} />}
        {page === "scan" && <ScanPage onNav={setPage} details={scanDetails} onDetailsChange={setScanDetails} onScanComplete={setScanId} />}
        {page === "analysis" && <AnalysisPage onNav={setPage} details={scanDetails} scanId={scanId} />}
        {page === "reports" && <ReportsPage details={scanDetails} scanId={scanId} />}
        {page === "patients" && <PatientsPage onNav={setPage} onViewReport={(scanId) => { setScanId(scanId); setPage("reports"); }} onNewScan={(patient) => {
          setScanDetails({
            fullName: patient.name,
            dob: patient.date_of_birth || "",
            eye: "Left Eye (OS)",
            gender: patient.gender || "",
            mobile: patient.mobile || "",
            notes: patient.notes || "",
            patient_id: patient.id,
          });
          setPage("scan");
        }} />}
        {page === "settings" && <SettingsPage />}
      </AppShell>
    </>
  );
}
