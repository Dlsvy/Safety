// 홈 화면 설치용 최소 서비스 워커 — 앱 화면만 저장(촬영·업로드는 항상 PC 와 직접 통신)
const C = "p3d-m-v1";
self.addEventListener("install", (e) => { e.waitUntil(caches.open(C).then((c) => c.addAll(["/m", "/m/icon.svg"]))); self.skipWaiting(); });
self.addEventListener("activate", (e) => { e.waitUntil(self.clients.claim()); });
self.addEventListener("fetch", (e) => {
  const u = new URL(e.request.url);
  if (e.request.method !== "GET" || u.pathname !== "/m") return;
  e.respondWith(fetch(e.request).then((r) => { const k = r.clone(); caches.open(C).then((c) => c.put("/m", k)); return r; })
    .catch(() => caches.match("/m")));
});
