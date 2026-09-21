const BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000/api";

let _token: string | null = null;

export function setToken(t: string | null) { _token = t; }
export function getToken() { return _token; }

export const AUTH_EXPIRED_EVENT = "rs:auth-expired";

function handleUnauthorized() {
  _token = null;
  localStorage.removeItem("rs_token");
  localStorage.removeItem("rs_user");
  window.dispatchEvent(new Event(AUTH_EXPIRED_EVENT));
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (_token) headers["Authorization"] = `Bearer ${_token}`;
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (res.status === 401 || res.status === 403) {
    handleUnauthorized();
    throw new Error("Session expired. Please sign in again.");
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

async function upload<T>(path: string, formData: FormData): Promise<T> {
  const headers: Record<string, string> = {};
  if (_token) headers["Authorization"] = `Bearer ${_token}`;
  const res = await fetch(`${BASE}${path}`, { method: "POST", headers, body: formData });
  if (res.status === 401 || res.status === 403) {
    handleUnauthorized();
    throw new Error("Session expired. Please sign in again.");
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Upload failed");
  }
  return res.json();
}

// ─── Auth ─────────────────────────────────────────────────────────────────────
export const authApi = {
  login: (email: string, password: string) =>
    request<{ token: string; user: { name: string; role: string; initials: string; email: string } }>(
      "POST", "/auth/login", { email, password }
    ),
  register: (data: { first_name: string; last_name: string; email: string; password: string }) =>
    request<{ message: string }>("POST", "/auth/register", data),
  forgotPassword: (email: string) =>
    request<{ message: string }>("POST", "/auth/forgot-password", { email }),
  requestAccess: (data: { name: string; email: string; institution: string; reason?: string }) =>
    request<{ message: string }>("POST", "/auth/request-access", data),
};

// ─── Dashboard ────────────────────────────────────────────────────────────────
export const dashboardApi = {
  getStats: () =>
    request<{
      kpis: { title: string; value: string; change: string }[];
      scan_activity: { day: string; scans: number; analyzed: number }[];
      disease_distribution: { name: string; value: number; color: string }[];
      ai_performance: { month: string; accuracy: number; f1: number }[];
      recent_activity: { time: string; action: string; patient: string; result: string; type: string }[];
    }>("GET", "/dashboard/stats"),
};

// ─── Patients ─────────────────────────────────────────────────────────────────
export const patientsApi = {
  list: (params?: {
    search?: string; status?: string; diagnosis?: string; risk?: string;
    date_from?: string; date_to?: string; page?: number; page_size?: number;
  }) => {
    const q = new URLSearchParams();
    if (params) Object.entries(params).forEach(([k, v]) => { if (v) q.set(k, String(v)); });
    return request<{
      patients: {
        id: string; name: string; age: number; condition: string; severity: string;
        lastScan: string; status: string; risk: string; contact: string; email: string;
      }[];
      total: number; filtered: number;
    }>("GET", `/patients?${q}`);
  },
  get: (id: string) =>
    request<{
      patient: { id: string; name: string; age: number; condition: string; severity: string; lastScan: string; status: string; risk: string; contact: string; email: string } | null;
      details: { patientId: string; fullName: string; dob: string; eye: string; gender: string; physician: string; notes: string } | null;
      scan_history: { date: string; type: string; result: string; confidence: string; status: string }[];
    }>("GET", `/patients/${id}`),
  create: (data: { name: string; age: number; contact?: string; email?: string; condition?: string; status?: string }) =>
    request<{
      id: string; name: string; age: number; condition: string; severity: string;
      lastScan: string; status: string; risk: string; contact: string; email: string;
    }>("POST", "/patients", data),
  delete: (id: string) => request<void>("DELETE", `/patients/${id}`),
  downloadReportPdf: async (patientId: string): Promise<Blob> => {
    const headers: Record<string, string> = {};
    if (_token) headers["Authorization"] = `Bearer ${_token}`;
    const res = await fetch(`${BASE}/patients/${encodeURIComponent(patientId)}/report/pdf`, { headers });
    if (res.status === 401 || res.status === 403) {
      handleUnauthorized();
      throw new Error("Session expired. Please sign in again.");
    }
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "PDF download failed");
    }
    return res.blob();
  },
};

// ─── Scans ─────────────────────────────────────────────────────────────────────
export const scansApi = {
  create: (data: {
    patient_name: string; patient_dob?: string;
    patient_gender?: string; patient_eye?: string;
    patient_mobile?: string; patient_notes?: string; image?: File;
    patient_id?: string;
  }) => {
    const fd = new FormData();
    Object.entries(data).forEach(([k, v]) => { if (v !== undefined) fd.append(k, v instanceof File ? v : String(v)); });
    return upload<{ scan_id: string; status: string; progress: number }>("/scans", fd);
  },
  getStatus: (scanId: string) =>
    request<{ scan_id: string; status: string; progress: number }>("GET", `/scans/${scanId}/status`),
  getAnalysis: (scanId: string) =>
    request<{
      patient: { patient_id: string; full_name: string; dob: string; gender: string; eye: string; physician: string };
      diagnosis: { primary: string; detail: string; confidence: number; risk_score: number; severity: string; icd10: string; etdrs_grade: string };
      probabilities: { name: string; pct: number; color: string }[];
      findings: { finding: string; status: string; severity: string }[];
      recommendations: { icon: string; title: string; desc: string; color: string }[];
      heatmap_available: boolean;
      image_url?: string | null;
      heatmap_url?: string | null;
    }>("GET", `/scans/${scanId}/analysis`),
  getReport: (scanId: string) =>
    request<{
      report_id: string; generated_date: string;
      patient_info: { patient_name: string; patient_id: string; dob: string; gender: string; mrn: string };
      imaging: { eye: string; image_quality: string; camera: string; fov: string; physician: string };
      diagnosis: { diagnosis: string; icd10: string; etdrs_grade: string; confidence: number; summary: string };
      findings: { finding: string; status: string; confidence: string; significance: string }[];
      recommendations: { priority: string; text: string }[];
    }>("GET", `/scans/${scanId}/report`),
};

// ─── MATLAB image-analysis add-on (independent of disease inference) ──────────
export interface MatlabQualityResult {
  quality_score: number;
  brightness_score: number;
  contrast_score: number;
  sharpness_score: number;
  fov_score: number;
  status: string;
}

export interface MatlabFeatureResult {
  mean_intensity?: number;
  contrast?: number;
  bright_region_percentage?: number;
  dark_region_percentage?: number;
  vessel_density?: number;
  retinal_field_area?: number;
  [key: string]: number | undefined;
}

export interface MatlabAnalysisResult {
  matlab_available: boolean;
  matlab_status?: string;
  quality?: MatlabQualityResult | null;
  enhanced_image?: string | null;
  features?: MatlabFeatureResult | null;
}

export const matlabApi = {
  analyze: (scanId: string) => {
    const fd = new FormData();
    fd.append("scan_id", scanId);
    return upload<MatlabAnalysisResult>("/matlab/analyze", fd);
  },
};

// ─── Analysis (latest) ────────────────────────────────────────────────────────
export const analysisApi = {
  getLatest: () => request<any>("GET", "/analysis/latest"),
};

// ─── Reports (latest) ─────────────────────────────────────────────────────────
export const reportsApi = {
  getLatest: () => request<any>("GET", "/reports/latest"),
  downloadPdf: async (scanId?: string | null): Promise<Blob> => {
    const headers: Record<string, string> = {};
    if (_token) headers["Authorization"] = `Bearer ${_token}`;
    const res = await fetch(
      scanId ? `${BASE}/scans/${encodeURIComponent(scanId)}/report/pdf` : `${BASE}/reports/latest/pdf`,
      { headers },
    );
    if (res.status === 401 || res.status === 403) {
      handleUnauthorized();
      throw new Error("Session expired. Please sign in again.");
    }
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "PDF download failed");
    }
    return res.blob();
  },
};

// ─── Analytics ────────────────────────────────────────────────────────────────
export const analyticsApi = {
  getSummary: () =>
    request<{
      summary: { title: string; value: string; change: string; trend: string }[];
      disease_trends: { month: string; dr: number; glaucoma: number; amd: number }[];
      model_performance: { month: string; accuracy: number; f1: number }[];
      scan_volume: { day: string; scans: number; analyzed: number }[];
      category_stats: { title: string; items: { label: string; pct: number; color: string }[] }[];
    }>("GET", "/analytics/summary"),
};

// ─── Settings ─────────────────────────────────────────────────────────────────
export const settingsApi = {
  getProfile: () => request<any>("GET", "/settings/profile"),
  updateProfile: (data: any) => request<any>("PUT", "/settings/profile", data),
  changePassword: (data: { current_password: string; new_password: string }) =>
    request<any>("PUT", "/settings/password", data),
  getNotifications: () => request<any>("GET", "/settings/notifications"),
  updateNotifications: (data: any) => request<any>("PUT", "/settings/notifications", data),
  updateTheme: (darkMode: boolean) => request<any>("PUT", "/settings/theme", { dark_mode: darkMode }),
};

// ─── Images (authenticated blob fetch) ─────────────────────────────────────────
export const imagesApi = {
  get: async (name: string): Promise<Blob> => {
    const headers: Record<string, string> = {};
    if (_token) headers["Authorization"] = `Bearer ${_token}`;
    const res = await fetch(`${BASE}/images/${encodeURIComponent(name)}`, { headers });
    if (res.status === 401 || res.status === 403) {
      handleUnauthorized();
      throw new Error("Session expired. Please sign in again.");
    }
    if (!res.ok) throw new Error("Image could not be loaded");
    return res.blob();
  },
};
