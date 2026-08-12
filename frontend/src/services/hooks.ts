import { useState, useEffect, useCallback, useRef } from "react";
import { setToken as setApiToken, authApi, dashboardApi, patientsApi, scansApi, settingsApi, imagesApi } from "./api";

// ─── Helpers ───────────────────────────────────────────────────────────────────
function useAsync<T>(fn: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    setLoading(true);
    setError(null);
    fn()
      .then((d) => { if (mounted.current) setData(d); })
      .catch((e) => { if (mounted.current) setError(e.message); })
      .finally(() => { if (mounted.current) setLoading(false); });
    return () => { mounted.current = false; };
  }, deps);

  return { data, loading, error, refetch: () => { setLoading(true); fn().then(setData).catch((e) => setError(e.message)).finally(() => setLoading(false)); } };
}

// ─── Auth ──────────────────────────────────────────────────────────────────────
export function useAuth() {
  const [user, setUser] = useState<{ name: string; role: string; initials: string; email: string } | null>(() => {
    try {
      const saved = localStorage.getItem("rs_user");
      return saved ? JSON.parse(saved) : null;
    } catch { return null; }
  });
  const [token, setTokenState] = useState<string | null>(() => {
    const saved = localStorage.getItem("rs_token");
    if (saved) setApiToken(saved);
    return saved;
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const login = useCallback(async (email: string, password: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await authApi.login(email, password);
      setApiToken(res.token);
      setTokenState(res.token);
      setUser(res.user);
      localStorage.setItem("rs_token", res.token);
      localStorage.setItem("rs_user", JSON.stringify(res.user));
      return true;
    } catch (e: any) {
      setError(e.message);
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  const logout = useCallback(() => {
    setApiToken(null);
    setTokenState(null);
    setUser(null);
    localStorage.removeItem("rs_token");
    localStorage.removeItem("rs_user");
  }, []);

  return { user, token, isAuthenticated: !!token, loading, error, login, logout };
}

// ─── Dashboard ─────────────────────────────────────────────────────────────────
export function useDashboard() {
  const { data, loading, error, refetch } = useAsync(() => dashboardApi.getStats(), []);
  return {
    kpis: data?.kpis ?? null,
    scanActivity: data?.scan_activity ?? null,
    diseaseDistribution: data?.disease_distribution ?? null,
    aiPerformance: data?.ai_performance ?? null,
    recentActivity: data?.recent_activity ?? null,
    loading, error, refetch,
  };
}

// ─── Patients ──────────────────────────────────────────────────────────────────
export function usePatients() {
  const [patients, setPatients] = useState<any[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const mounted = useRef(true);

  const fetchList = useCallback(async (params?: Parameters<typeof patientsApi.list>[0]) => {
    setLoading(true);
    setError(null);
    try {
      const res = await patientsApi.list(params);
      if (mounted.current) { setPatients(res.patients); setTotal(res.filtered); }
    } catch (e: any) {
      if (mounted.current) setError(e.message);
    } finally {
      if (mounted.current) setLoading(false);
    }
  }, []);

  useEffect(() => { mounted.current = true; fetchList(); return () => { mounted.current = false; }; }, [fetchList]);

  const create = useCallback(async (data: { name: string; age: number; contact?: string; email?: string; condition?: string; status?: string }) => {
    const p = await patientsApi.create(data);
    setPatients((l) => [p, ...l]);
    return p;
  }, []);

  const remove = useCallback(async (id: string) => {
    await patientsApi.delete(id);
    setPatients((l) => l.filter((x) => x.id !== id));
  }, []);

  return { patients, total, loading, error, refetch: fetchList, create, remove };
}

// ─── Scan ──────────────────────────────────────────────────────────────────────
export function useScan() {
  const [scanId, setScanId] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("");
  const [progress, setProgress] = useState(0);
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval>>();
  const animRef = useRef<ReturnType<typeof setInterval>>();

  const startScan = useCallback(async (data: {
    patient_name: string; patient_dob?: string;
    patient_gender?: string; patient_eye?: string;
    patient_mobile?: string; patient_notes?: string; image?: File;
    patient_id?: string;
  }) => {
    setScanning(true);
    setError(null);
    setProgress(0);

    const scanStartTime = Date.now();
    const MIN_SCAN_DURATION = 6000;

    let animProgress = 0;
    const steps = [3, 8, 15, 22, 28, 35, 42, 48, 55, 62, 68, 75, 82, 88, 95, 100];
    let stepIdx = 0;
    animRef.current = setInterval(() => {
      if (stepIdx < steps.length) {
        animProgress = steps[stepIdx];
        stepIdx++;
      }
      setProgress(Math.round(animProgress));
    }, 375);

    let backendResult: { scan_id: string; status: string } | null = null;

    const finishScan = () => {
      clearInterval(animRef.current);
      clearInterval(pollRef.current);
      setProgress(100);
      setTimeout(() => setScanning(false), 800);
    };

    try {
      const res = await scansApi.create(data);
      setScanId(res.scan_id);
      setStatus(res.status);
      backendResult = res;

      pollRef.current = setInterval(async () => {
        try {
          const s = await scansApi.getStatus(res.scan_id);
          setStatus(s.status);
          if (s.status === "completed" || s.status === "failed") {
            clearInterval(pollRef.current);
            if (s.status === "failed") setError("Scan analysis failed");

            const elapsed = Date.now() - scanStartTime;
            if (elapsed >= MIN_SCAN_DURATION) {
              finishScan();
            } else {
              setTimeout(finishScan, MIN_SCAN_DURATION - elapsed);
            }
          }
        } catch {
          clearInterval(pollRef.current);
          const elapsed = Date.now() - scanStartTime;
          if (elapsed >= MIN_SCAN_DURATION) {
            finishScan();
          } else {
            setError("Status check failed");
            setTimeout(finishScan, MIN_SCAN_DURATION - elapsed);
          }
        }
      }, 800);

      return res.scan_id;
    } catch (e: any) {
      clearInterval(animRef.current);
      clearInterval(pollRef.current);
      setError(e.message);
      setScanning(false);
      return null;
    }
  }, []);

  const reset = useCallback(() => {
    clearInterval(animRef.current);
    clearInterval(pollRef.current);
    setScanId(null);
    setStatus("");
    setProgress(0);
    setScanning(false);
    setError(null);
    if (pollRef.current) clearInterval(pollRef.current);
  }, []);

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  return { scanId, status, progress, scanning, error, startScan, reset };
}

// ─── Analysis ───────────────────────────────────────────────────────────────────
export function useAnalysis(scanId: string | null) {
  const [analysis, setAnalysis] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!scanId) return;
    setLoading(true);
    scansApi.getAnalysis(scanId)
      .then(setAnalysis)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [scanId]);

  return { analysis, loading, error };
}

// ─── Authenticated image (blob → object URL) ─────────────────────────────────────
export function useBlobImage(name: string | null | undefined) {
  const [src, setSrc] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let objectUrl: string | null = null;
    setSrc(null);
    setError(null);
    if (!name) return;
    imagesApi.get(name)
      .then((blob) => {
        if (cancelled) return;
        objectUrl = URL.createObjectURL(blob);
        setSrc(objectUrl);
      })
      .catch((e) => { if (!cancelled) setError(e.message); });
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [name]);

  return { src, error };
}

// ─── Report ────────────────────────────────────────────────────────────────────
export function useReport(scanId: string | null) {
  const [report, setReport] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!scanId) return;
    setLoading(true);
    scansApi.getReport(scanId)
      .then(setReport)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [scanId]);

  return { report, loading, error };
}

// ─── Settings ──────────────────────────────────────────────────────────────────
export function useSettings() {
  const { data: profile, loading: profileLoading, error: profileError, refetch: refetchProfile } = useAsync(() => settingsApi.getProfile(), []);
  const { data: notifications, loading: notifLoading, error: notifError, refetch: refetchNotif } = useAsync(() => settingsApi.getNotifications(), []);

  const updateProfile = useCallback((data: any) => settingsApi.updateProfile(data), []);
  const changePassword = useCallback((current_password: string, new_password: string) => settingsApi.changePassword({ current_password, new_password }), []);
  const updateNotifications = useCallback((data: any) => settingsApi.updateNotifications(data), []);
  const toggleTheme = useCallback((dark: boolean) => settingsApi.updateTheme(dark), []);

  return { profile, notifications, loading: profileLoading || notifLoading, error: profileError || notifError, updateProfile, changePassword, updateNotifications, toggleTheme, refetchProfile, refetchNotif };
}
