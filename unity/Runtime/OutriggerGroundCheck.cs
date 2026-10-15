// 아웃트리거(지지대) 설치 지점의 지반 높이·경사를 사진측량 지형에서 읽어 판정.
using UnityEngine;

namespace Photo3D
{
    public class OutriggerGroundCheck : MonoBehaviour
    {
        public Photo3DSite site;
        public Transform[] pads = new Transform[4];   // 아웃트리거 패드 위치(수평 위치만 사용)
        [Tooltip("허용 경사(도) — 장비 제원 확인")] public float maxSlopeDeg = 1.0f;
        [Tooltip("패드 간 허용 높이차(m)")] public float maxHeightDiff = 0.10f;
        public float[] groundHeights = new float[4];
        public float slopeDeg;
        public bool ok;

        [ContextMenu("지반 판정 실행")]
        public void Evaluate()
        {
            LayerMask mask = site ? site.SiteMask : (LayerMask)1;
            var pts = new Vector3[pads.Length];
            for (int i = 0; i < pads.Length; i++)
            {
                var o = pads[i].position + Vector3.up * 50f;
                if (Physics.Raycast(o, Vector3.down, out var h, 200f, mask, QueryTriggerInteraction.Ignore))
                {
                    groundHeights[i] = h.point.y;
                    pts[i] = h.point;
                }
                else
                {
                    groundHeights[i] = float.NaN;
                    Debug.LogWarning($"[Photo3D] 패드 {i} 아래에 지형 데이터 없음 — 해당 위치 재촬영 필요");
                }
            }
            // 최소제곱 평면 → 경사
            Vector3 c = Vector3.zero; int n = 0;
            foreach (var p in pts) if (!float.IsNaN(p.y)) { c += p; n++; }
            if (n < 3) { ok = false; return; }
            c /= n;
            float sxx = 0, sxz = 0, szz = 0, sxy = 0, szy = 0;
            foreach (var p in pts)
            {
                if (float.IsNaN(p.y)) continue;
                var d = p - c;
                sxx += d.x * d.x; sxz += d.x * d.z; szz += d.z * d.z; sxy += d.x * d.y; szy += d.z * d.y;
            }
            float det = sxx * szz - sxz * sxz;
            float gx = (sxy * szz - szy * sxz) / det, gz = (szy * sxx - sxy * sxz) / det;
            slopeDeg = Mathf.Atan(Mathf.Sqrt(gx * gx + gz * gz)) * Mathf.Rad2Deg;
            float mn = float.MaxValue, mx = float.MinValue;
            foreach (var y in groundHeights) if (!float.IsNaN(y)) { mn = Mathf.Min(mn, y); mx = Mathf.Max(mx, y); }
            ok = slopeDeg <= maxSlopeDeg && (mx - mn) <= maxHeightDiff;
            Debug.Log($"[Photo3D] 아웃트리거 지반: 경사 {slopeDeg:0.00}°, 높이차 {(mx - mn) * 1000:0}mm → {(ok ? "적합" : "부적합")}");
        }
    }
}
