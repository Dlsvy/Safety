// photo3d 로 만든 현장 모델을 유니티 물리에 연결한다.
// - visualRoot   : site_model.glb (텍스처, 보기용)
// - colliderRoot : site_collider.glb (경량, 충돌 판정용) → MeshCollider 자동 부착, 렌더러 끔
// 1 unit = 1 m, +Y = 위. 0번 ArUco 마커 중심이 원점(또는 GCP 현장좌표 - offset).
using UnityEngine;

namespace Photo3D
{
    public class Photo3DSite : MonoBehaviour
    {
        public GameObject visualRoot;
        public GameObject colliderRoot;
        [Tooltip("site_estimated.glb — Poisson 으로 메운 '추정 영역'(측정 아님). 충돌은 하되 판정 시 경고")]
        public GameObject estimatedRoot;
        public string estimatedLayer = "SiteEstimated";
        [Tooltip("현장 메쉬가 들어갈 레이어 이름(Project Settings > Tags and Layers 에 추가)")]
        public string siteLayer = "Site";
        public bool hideColliderMesh = true;

        void Awake() { Setup(); }

        public void Setup()
        {
            int layer = LayerMask.NameToLayer(siteLayer);
            if (layer < 0)
            {
                Debug.LogWarning($"[Photo3D] 레이어 '{siteLayer}' 가 없어 Default 를 씁니다.");
                layer = 0;
            }
            AddColliders(colliderRoot, layer, hideColliderMesh);
            int el = LayerMask.NameToLayer(estimatedLayer);
            if (estimatedRoot != null)
            {
                if (el < 0) Debug.LogWarning($"[Photo3D] 레이어 '{estimatedLayer}' 가 없어 추정 영역을 구분할 수 없습니다.");
                AddColliders(estimatedRoot, el < 0 ? layer : el, false); // 추정 영역은 주황으로 보이게 둔다
            }
        }

        static void AddColliders(GameObject root, int layer, bool hide)
        {
            if (root == null) return;
            foreach (var mf in root.GetComponentsInChildren<MeshFilter>(true))
            {
                mf.gameObject.layer = layer;
                var mc = mf.GetComponent<MeshCollider>();
                if (mc == null) mc = mf.gameObject.AddComponent<MeshCollider>(); // ?? 는 유니티 가짜 null 을 못 거른다
                mc.sharedMesh = mf.sharedMesh;
                mc.convex = false; // 지형·건물은 오목하므로 비볼록 유지
                if (hide)
                {
                    var r = mf.GetComponent<MeshRenderer>();
                    if (r) r.enabled = false;
                }
            }
        }

        public LayerMask SiteMask
        {
            get
            {
                int l = LayerMask.NameToLayer(siteLayer), e = LayerMask.NameToLayer(estimatedLayer);
                int m = l < 0 ? 1 : (1 << l);
                if (e >= 0) m |= 1 << e;
                return m;
            }
        }

        public bool IsEstimated(Collider c) =>
            c != null && c.gameObject.layer == LayerMask.NameToLayer(estimatedLayer);
    }
}
