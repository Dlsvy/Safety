// 크레인 붐·후크·인양물과 현장(사진측량 메쉬) 사이 최소 이격거리 감시.
// 비볼록 MeshCollider 는 Physics.ClosestPoint 를 지원하지 않으므로,
// 캡슐 반경을 키워 가며 OverlapCapsule 로 이분 탐색해 이격거리를 구한다(정밀도 ~1cm).
using System.Collections.Generic;
using UnityEngine;

namespace Photo3D
{
    [System.Serializable]
    public class Segment
    {
        public string name = "boom";
        public Transform a;           // 시작점 (예: 붐 풋핀)
        public Transform b;           // 끝점 (예: 붐 팁 / 후크 / 인양물 하단)
        public float radius = 0.4f;   // 부재 반경(m)
        [HideInInspector] public float clearance;
        [HideInInspector] public Collider closest;
    }

    public enum ClearanceLevel { Safe, Warning, Danger }

    public class CraneClearanceMonitor : MonoBehaviour
    {
        public Photo3DSite site;
        public List<Segment> segments = new List<Segment>();
        [Tooltip("이격거리 경고 기준(m)")] public float warningDistance = 3.0f;
        [Tooltip("이격거리 위험 기준(m)")] public float dangerDistance = 1.0f;
        [Tooltip("붐이 지면에 붙는 붐 풋 부근은 판정에서 제외할 길이(m)")] public float ignoreNearA = 1.5f;
        public LayerMask extraMask; // 다른 장비·가설구조물 등
        public ClearanceLevel level;
        public float minClearance;
        [Tooltip("가장 가까운 면이 Poisson 추정 영역(측정 아님)인지")] public bool closestIsEstimated;
        public event System.Action<Segment, ClearanceLevel> OnLevelChanged;

        readonly Collider[] buf = new Collider[64];

        void FixedUpdate()
        {
            LayerMask mask = (site ? site.SiteMask : (LayerMask)1) | extraMask;
            var prev = level;
            minClearance = float.PositiveInfinity;
            Segment worst = null;
            foreach (var s in segments)
            {
                if (!s.a || !s.b) continue;
                Vector3 p0 = s.a.position, p1 = s.b.position;
                Vector3 dir = (p1 - p0);
                float len = dir.magnitude;
                if (len < 1e-3f) continue;
                p0 += dir / len * Mathf.Min(ignoreNearA, len * 0.5f);
                s.clearance = Clearance(p0, p1, s.radius, mask, out s.closest);
                if (s.clearance < minClearance) { minClearance = s.clearance; worst = s; }
            }
            closestIsEstimated = worst != null && site != null && site.IsEstimated(worst.closest);
            level = minClearance <= dangerDistance ? ClearanceLevel.Danger
                  : minClearance <= warningDistance ? ClearanceLevel.Warning : ClearanceLevel.Safe;
            if (level != prev && worst != null) OnLevelChanged?.Invoke(worst, level);
        }

        float Clearance(Vector3 p0, Vector3 p1, float r, LayerMask mask, out Collider hit)
        {
            hit = null;
            float search = warningDistance * 2f;
            if (Count(p0, p1, r + search, mask, out _) == 0) return float.PositiveInfinity;
            if (Count(p0, p1, r, mask, out hit) > 0) return 0f; // 이미 접촉
            float lo = 0f, hi = search;
            for (int i = 0; i < 10; i++)
            {
                float mid = 0.5f * (lo + hi);
                if (Count(p0, p1, r + mid, mask, out var h) > 0) { hi = mid; hit = h; } else lo = mid;
            }
            return hi;
        }

        int Count(Vector3 p0, Vector3 p1, float r, LayerMask mask, out Collider first)
        {
            int n = Physics.OverlapCapsuleNonAlloc(p0, p1, r, buf, mask, QueryTriggerInteraction.Ignore);
            int m = 0; first = null;
            for (int i = 0; i < n; i++)
            {
                if (buf[i].transform.IsChildOf(transform)) continue; // 크레인 자신 제외
                if (first == null) first = buf[i];
                m++;
            }
            return m;
        }

        void OnDrawGizmos()
        {
            foreach (var s in segments)
            {
                if (!s.a || !s.b) continue;
                Gizmos.color = s.clearance <= dangerDistance ? Color.red
                             : s.clearance <= warningDistance ? Color.yellow : Color.green;
                Gizmos.DrawLine(s.a.position, s.b.position);
                Gizmos.DrawWireSphere(s.b.position, s.radius);
            }
        }

        void OnGUI()
        {
            var st = new GUIStyle(GUI.skin.box) { fontSize = 18, alignment = TextAnchor.MiddleLeft };
            st.normal.textColor = level == ClearanceLevel.Danger ? Color.red
                                : level == ClearanceLevel.Warning ? Color.yellow : Color.green;
            string d = float.IsInfinity(minClearance) ? $"> {warningDistance * 2:0.0} m" : $"{minClearance:0.00} m";
            GUI.Box(new Rect(10, 10, 360, 34), $"크레인 최소 이격 {d}  [{level}]", st);
            if (closestIsEstimated && level != ClearanceLevel.Safe)
            {
                var w = new GUIStyle(st); w.normal.textColor = new Color(1f, 0.55f, 0f);
                GUI.Box(new Rect(10, 48, 360, 34), "주의: 추정 영역(실측 아님) 기준 판정", w);
            }
        }
    }
}
