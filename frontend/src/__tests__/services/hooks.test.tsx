import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import { useAuth, useDashboard, usePatients } from "../../services/hooks";

const API_BASE = "http://127.0.0.1:8000/api";

function mockFetch(status: number, body: unknown, ok?: boolean) {
  return vi.fn().mockResolvedValue({
    ok: ok ?? (status >= 200 && status < 300),
    status,
    json: () => Promise.resolve(body),
  });
}

beforeEach(() => {
  vi.restoreAllMocks();
  localStorage.clear();
});

describe("useAuth", () => {
  it("starts unauthenticated", () => {
    const { result } = renderHook(() => useAuth());
    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.user).toBeNull();
  });

  it("login succeeds and stores token", async () => {
    global.fetch = mockFetch(200, {
      token: "test-token-123",
      user: { name: "Dr. Test", role: "Ophthalmologist", initials: "DT", email: "test@test.com" },
    });

    const { result } = renderHook(() => useAuth());
    let ok: boolean = false;
    await act(async () => { ok = await result.current.login("test@test.com", "pass"); });

    expect(ok).toBe(true);
    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.user?.name).toBe("Dr. Test");
    expect(localStorage.getItem("rs_token")).toBe("test-token-123");
  });

  it("login fails on bad credentials", async () => {
    global.fetch = mockFetch(401, { detail: "Invalid email or password" }, false);

    const { result } = renderHook(() => useAuth());
    let ok: boolean = false;
    await act(async () => { ok = await result.current.login("bad@test.com", "wrong"); });

    expect(ok).toBe(false);
    expect(result.current.isAuthenticated).toBe(false);
  });

  it("logout clears token and user", async () => {
    global.fetch = mockFetch(200, {
      token: "tok", user: { name: "Dr. A", role: "Doc", initials: "DA", email: "a@b.com" },
    });

    const { result } = renderHook(() => useAuth());
    await act(async () => { await result.current.login("a@b.com", "p"); });
    expect(result.current.isAuthenticated).toBe(true);

    act(() => result.current.logout());
    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.user).toBeNull();
    expect(localStorage.getItem("rs_token")).toBeNull();
  });

  it("restores session from localStorage", () => {
    localStorage.setItem("rs_token", "saved-token");
    localStorage.setItem("rs_user", JSON.stringify({ name: "Saved User", role: "Admin", initials: "SU", email: "saved@test.com" }));

    const { result } = renderHook(() => useAuth());
    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.user?.name).toBe("Saved User");
  });
});

describe("useDashboard", () => {
  it("fetches and returns dashboard data", async () => {
    const mockData = {
      kpis: [{ title: "Total Scans", value: "100", change: "+5%" }],
      scan_activity: [{ day: "Mon", scans: 10, analyzed: 8 }],
      disease_distribution: [{ name: "DR", value: 34, color: "#2563EB" }],
      ai_performance: [{ month: "Jan", accuracy: 95, f1: 93 }],
      recent_activity: [{ time: "2m ago", action: "Scan", patient: "John", result: "OK", type: "success" }],
    };

    global.fetch = mockFetch(200, mockData);

    const { result } = renderHook(() => useDashboard());

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.kpis).toEqual(mockData.kpis);
    expect(result.current.scanActivity).toEqual(mockData.scan_activity);
    expect(result.current.diseaseDistribution).toEqual(mockData.disease_distribution);
    expect(result.current.aiPerformance).toEqual(mockData.ai_performance);
    expect(result.current.recentActivity).toEqual(mockData.recent_activity);
  });

  it("handles fetch error", async () => {
    global.fetch = mockFetch(500, { detail: "Server error" }, false);

    const { result } = renderHook(() => useDashboard());

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.error).toBeTruthy();
  });
});

describe("usePatients", () => {
  const mockPatients = {
    patients: [
      { id: "P-001", name: "Alice", age: 50, condition: "Healthy", severity: "None", lastScan: "2024-01-01", status: "Stable", risk: "Low", contact: "", email: "" },
    ],
    total: 1,
    filtered: 1,
  };

  it("fetches and returns patients", async () => {
    global.fetch = mockFetch(200, mockPatients);

    const { result } = renderHook(() => usePatients());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
      expect(result.current.patients).toHaveLength(1);
    });
    expect(result.current.patients[0].name).toBe("Alice");
  });

  it("creates a patient", async () => {
    global.fetch = vi.fn()
      .mockResolvedValueOnce({ ok: true, status: 200, json: () => Promise.resolve({ patients: [], total: 0, filtered: 0 }) })
      .mockResolvedValueOnce({ ok: true, status: 201, json: () => Promise.resolve({ id: "P-002", name: "Bob", age: 40, condition: "Glaucoma", severity: "Mild", lastScan: "2024-01-01", status: "Active", risk: "Medium", contact: "", email: "" }) });

    const { result } = renderHook(() => usePatients());
    await waitFor(() => expect(result.current.loading).toBe(false));

    let p: any;
    await act(async () => { p = await result.current.create({ name: "Bob", age: 40, condition: "Glaucoma" }); });

    expect(p.id).toBe("P-002");
    expect(p.name).toBe("Bob");
    expect(result.current.patients).toHaveLength(1);
  });
});
