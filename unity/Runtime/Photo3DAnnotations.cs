// photo3d 앱에서 그린 '현장 표시'(annotations.json)와 촬영 위치(cameras.json)를 유니티 씬에 만든다.
// - 거리: 선 + 이름표,  반경(선회반경·낙하 위험반경): 바닥 원,  구역: 트리거 BoxCollider(레이어 SiteZone),  위험 표시: 기둥 + 공
// 좌표: 파일은 glTF(Y-up, 오른손, m). glTFast 와 같게 X 를 뒤집어 유니티(왼손) 좌표로 바꾼다.
using System;
using System.Collections.Generic;
using UnityEngine;

namespace Photo3D
{
    [Serializable] public class AnnItem
    {
        public string id, type, label, memo, level;
        public float[] a, b, center, pos, size;
        public float radius, yaw;
    }
    [Serializable] public class AnnFile { public int version; public string frame; public AnnItem[] items; }
    [Serializable] public class CamItem { public string name; public float[] pos, fwd, up; }
    [Serializable] public class CamFile { public string frame; public CamItem[] cameras; }

    /// <summary>위험 표시·반경에 붙는 정보(시뮬레이션 스크립트가 읽어 씀)</summary>
    public class Photo3DMark : MonoBehaviour
    {
        public string kind, label, memo, level;
        public float radius;
    }

    public class Photo3DAnnotations : MonoBehaviour
    {
        public TextAsset annotationsJson, camerasJson;
        public string zoneLayer = "SiteZone";
        public bool showCamerasGizmo = true;
        readonly List<Vector3> camPos = new List<Vector3>();
        readonly List<Vector3> camFwd = new List<Vector3>();

        static Vector3 V(float[] a) => a == null || a.Length < 3 ? Vector3.zero : new Vector3(-a[0], a[1], a[2]);

        static Color LevelColor(string lv) => lv == "warn" ? new Color(0.98f, 0.70f, 0.10f)
            : lv == "critical" ? new Color(0.82f, 0.23f, 0.23f) : new Color(0.93f, 0.51f, 0.35f);

        static Material Mat(Color c, float alpha)
        {
            var sh = Shader.Find("Universal Render Pipeline/Unlit") ?? Shader.Find("Unlit/Color") ?? Shader.Find("Sprites/Default");
            var m = new Material(sh) { color = new Color(c.r, c.g, c.b, alpha) };
            return m;
        }

        static LineRenderer Line(Transform parent, string name, Vector3[] pts, Color c, bool loop = false)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var lr = go.AddComponent<LineRenderer>();
            lr.useWorldSpace = false;
            lr.loop = loop;
            lr.positionCount = pts.Length;
            lr.SetPositions(pts);
            lr.widthMultiplier = 0.05f;
            lr.material = Mat(c, 1f);
            lr.startColor = lr.endColor = c;
            return lr;
        }

        static void Label(Transform parent, string text, Vector3 p, Color c)
        {
            var go = new GameObject("Label");
            go.transform.SetParent(parent, false);
            go.transform.localPosition = p;
            var tm = go.AddComponent<TextMesh>();
            tm.text = text;
            tm.characterSize = 0.15f;
            tm.fontSize = 48;
            tm.anchor = TextAnchor.LowerCenter;
            tm.color = c;
        }

        [ContextMenu("Build")]
        public void Build()
        {
            for (int i = transform.childCount - 1; i >= 0; i--)
            {
                var ch = transform.GetChild(i);
                if (Application.isPlaying) Destroy(ch.gameObject); else DestroyImmediate(ch.gameObject);
            }
            if (annotationsJson != null)
            {
                var f = JsonUtility.FromJson<AnnFile>(annotationsJson.text);
                int zl = LayerMask.NameToLayer(zoneLayer);
                foreach (var it in f.items ?? Array.Empty<AnnItem>())
                {
                    var g = new GameObject($"{it.type}_{it.label}");
                    g.transform.SetParent(transform, false);
                    var mk = g.AddComponent<Photo3DMark>();
                    mk.kind = it.type; mk.label = it.label; mk.memo = it.memo; mk.level = it.level; mk.radius = it.radius;
                    switch (it.type)
                    {
                        case "distance":
                        {
                            var a = V(it.a); var b = V(it.b);
                            Line(g.transform, "Line", new[] { a, b }, new Color(0.49f, 0.77f, 1f));
                            Label(g.transform, $"{it.label} {Vector3.Distance(a, b):F2} m", (a + b) / 2, Color.white);
                            break;
                        }
                        case "circle":
                        {
                            var c = V(it.center);
                            var pts = new Vector3[96];
                            for (int k = 0; k < pts.Length; k++)
                            {
                                float t = k * Mathf.PI * 2 / pts.Length;
                                pts[k] = c + new Vector3(Mathf.Cos(t), 0.02f, Mathf.Sin(t)) * it.radius;
                            }
                            Line(g.transform, "Ring", pts, new Color(0.93f, 0.51f, 0.35f), true);
                            Label(g.transform, $"{it.label} R {it.radius:F2} m", c + Vector3.up * 0.5f, Color.white);
                            break;
                        }
                        case "box":
                        {
                            var cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
                            cube.name = "Zone";
                            cube.transform.SetParent(g.transform, false);
                            cube.transform.localPosition = V(it.center);
                            cube.transform.localRotation = Quaternion.Euler(0, -it.yaw, 0);  // glTF→유니티: X 반전 → 회전 방향 반대
                            cube.transform.localScale = new Vector3(it.size[0], it.size[1], it.size[2]);
                            cube.GetComponent<Renderer>().sharedMaterial = Mat(new Color(0.31f, 0.82f, 0.65f), 0.25f);
                            cube.GetComponent<BoxCollider>().isTrigger = true;  // 장비가 들어오면 OnTrigger 로 경고
                            if (zl >= 0) cube.layer = zl;
                            Label(g.transform, it.label, V(it.center) + Vector3.up * (it.size[1] / 2 + 0.3f), Color.white);
                            break;
                        }
                        case "pin":
                        {
                            var p = V(it.pos);
                            var col = LevelColor(it.level);
                            Line(g.transform, "Pole", new[] { p, p + Vector3.up * 2f }, col);
                            var s = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                            DestroyImmediate(s.GetComponent<Collider>());
                            s.transform.SetParent(g.transform, false);
                            s.transform.localPosition = p + Vector3.up * 2f;
                            s.transform.localScale = Vector3.one * 0.35f;
                            s.GetComponent<Renderer>().sharedMaterial = Mat(col, 1f);
                            Label(g.transform, it.label, p + Vector3.up * 2.3f, col);
                            break;
                        }
                    }
                }
            }
            camPos.Clear(); camFwd.Clear();
            if (camerasJson != null)
            {
                var cf = JsonUtility.FromJson<CamFile>(camerasJson.text);
                foreach (var c in cf.cameras ?? Array.Empty<CamItem>()) { camPos.Add(V(c.pos)); camFwd.Add(V(c.fwd)); }
            }
        }

        void OnDrawGizmosSelected()
        {
            if (!showCamerasGizmo) return;
            Gizmos.color = new Color(0.95f, 0.72f, 0.02f);
            Gizmos.matrix = transform.localToWorldMatrix;
            for (int i = 0; i < camPos.Count; i++)
            {
                Gizmos.DrawWireSphere(camPos[i], 0.1f);
                Gizmos.DrawLine(camPos[i], camPos[i] + camFwd[i].normalized * 0.5f);
            }
        }
    }
}
