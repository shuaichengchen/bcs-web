self.addEventListener("install", (event) => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

// 先不做离线缓存，避免新手踩坑
self.addEventListener("fetch", (event) => {});
