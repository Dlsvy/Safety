// 메뉴: Photo3D > Import Site Folder...
// photo3d 출력 폴더(site_model.glb, site_collider.glb, site_meta.json)를 Assets/Photo3DSites/<이름>/ 로 복사하고
// 씬에 배치 + Photo3DSite 구성. glTF 반입은 Unity glTFast 패키지(com.unity.cloud.gltfast) 필요.
using System.IO;
using UnityEditor;
using UnityEngine;

namespace Photo3D.EditorTools
{
    public static class Photo3DImporter
    {
        [MenuItem("Photo3D/Import Site Folder...")]
        public static void Import()
        {
            string src = EditorUtility.OpenFolderPanel("photo3d 출력 폴더 선택", "", "");
            if (string.IsNullOrEmpty(src)) return;
            string name = new DirectoryInfo(src).Name;
            string dstRel = $"Assets/Photo3DSites/{name}";
            Directory.CreateDirectory(dstRel);
            foreach (var f in new[] { "site_model.glb", "site_model_light.glb", "site_collider.glb", "site_estimated.glb", "site_filled.glb", "site_splat.ply",
                                       "site_meta.json", "report.json", "quality_report.html", "annotations.json", "cameras.json" })
            {
                var p = Path.Combine(src, f);
                if (File.Exists(p)) File.Copy(p, Path.Combine(dstRel, f), true);
            }
            AssetDatabase.Refresh();
            var root = BuildSite(dstRel, name);
            if (root == null) return;
            Undo.RegisterCreatedObjectUndo(root, "Import photo3d site");
            Selection.activeGameObject = root;
            WarnGrade(dstRel);
        }

        public static void WarnGrade(string dstRel)
        {
            var meta = Path.Combine(dstRel, "report.json");
            if (File.Exists(meta) && File.ReadAllText(meta).Contains("\"grade\": \"D\""))
                EditorUtility.DisplayDialog("Photo3D", "이 모델은 축척이 확정되지 않았습니다(등급 D). 안전 판정에 쓰지 마세요.", "확인");
        }

        static GameObject Load(string dstRel, params string[] names)
        {
            foreach (var n in names)
            {
                var g = AssetDatabase.LoadAssetAtPath<GameObject>($"{dstRel}/{n}");
                if (g != null) return g;
            }
            return null;
        }

        /// 현장 폴더의 모델로 장면 오브젝트를 만든다. OBJ(유니티 기본 지원, .unitypackage 로 온 경우) 우선, 없으면 GLB(glTFast).
        public static GameObject BuildSite(string dstRel, string name)
        {
            // 면이 많은 현장은 보기용 가벼운 모델(사진 텍스처 다시 구움)을 쓴다 — 충돌·측정은 site_collider
            var vis = Load(dstRel, "site_model.obj", "site_model_light.glb", "site_model.glb");
            var col = Load(dstRel, "site_collider.obj", "site_collider.glb");
            if (vis == null || col == null)
            {
                EditorUtility.DisplayDialog("Photo3D", "모델을 불러오지 못했습니다. GLB 만 있으면 Package Manager 에서 glTFast(com.unity.cloud.gltfast)를 설치하세요.", "확인");
                return null;
            }
            var root = new GameObject($"Site_{name}");
            var site = root.AddComponent<Photo3DSite>();
            site.visualRoot = (GameObject)PrefabUtility.InstantiatePrefab(vis, root.transform);
            site.colliderRoot = (GameObject)PrefabUtility.InstantiatePrefab(col, root.transform);
            var est = Load(dstRel, "site_estimated.obj", "site_estimated.glb");
            if (est != null) site.estimatedRoot = (GameObject)PrefabUtility.InstantiatePrefab(est, root.transform);
            site.Setup();
            // 앱에서 그린 현장 표시(거리·반경·구역·위험)와 촬영 위치
            var annJ = AssetDatabase.LoadAssetAtPath<TextAsset>($"{dstRel}/annotations.json");
            var camJ = AssetDatabase.LoadAssetAtPath<TextAsset>($"{dstRel}/cameras.json");
            if (annJ != null || camJ != null)
            {
                var ag = new GameObject("Annotations");
                ag.transform.SetParent(root.transform, false);
                var an = ag.AddComponent<Photo3DAnnotations>();
                an.annotationsJson = annJ;
                an.camerasJson = camJ;
                an.Build();
            }
            return root;
        }
    }
}
