// .unitypackage(photo3d 가 만든 현장 패키지)를 가져오면 자동으로 현장 프리팹을 만든다.
// Assets/Photo3DSites/<이름>/site_meta.json 이 새로 들어오면 → 같은 폴더에 Site_<이름>.prefab
// (보기 모델 + 숨긴 충돌 메쉬(MeshCollider) + 추정 영역 + 현장 표시). 장면에 끌어다 놓으면 끝.
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace Photo3D
{
    public class Photo3DAutoSetup : AssetPostprocessor
    {
        static readonly HashSet<string> pending = new HashSet<string>();

        static void OnPostprocessAllAssets(string[] imported, string[] deleted, string[] moved, string[] movedFrom)
        {
            foreach (var p in imported)
            {
                var n = p.Replace('\\', '/');
                if (n.StartsWith("Assets/Photo3DSites/") && n.EndsWith("/site_meta.json"))
                    pending.Add(Path.GetDirectoryName(n).Replace('\\', '/'));
            }
            if (pending.Count > 0) EditorApplication.delayCall += BuildPending;  // 모델 가져오기가 끝난 뒤에
        }

        static void BuildPending()
        {
            var dirs = new List<string>(pending);
            pending.Clear();
            foreach (var dir in dirs)
            {
                string name = Path.GetFileName(dir);
                string prefab = $"{dir}/Site_{name}.prefab";
                if (File.Exists(prefab)) continue;
                var root = Photo3DImporter.BuildSite(dir, name);
                if (root == null) continue;
                var saved = PrefabUtility.SaveAsPrefabAsset(root, prefab);
                Object.DestroyImmediate(root);
                Debug.Log($"[Photo3D] 현장 프리팹을 만들었습니다: {prefab} — 장면에 끌어다 놓으세요(1 unit = 1 m, +Y 위)");
                if (saved != null) { Selection.activeObject = saved; EditorGUIUtility.PingObject(saved); }
                Photo3DImporter.WarnGrade(dir);
            }
        }
    }
}
