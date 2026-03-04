using System.Collections.Generic;
using UnityEngine;

public class ProductPool : MonoBehaviour
{
    [Header("Pool Setup")]
    public GameObject prefab;  // 상자 프리팹
    public int prewarm = 10;   // 시작 시 미리 만들어둘 개수
    public Transform storage;  // 비활성 보관용 부모(없으면 자동 생성)

    private readonly Queue<GameObject> q = new();

    private void Awake()
    {
        if (!prefab)
        {
            // 프리팹이 없으면 더 이상 동작해도 의미가 없으니 비활성화
            enabled = false;
            return;
        }

        if (!storage)
        {
            // storage가 없으면 자동 생성 (Hierarchy 정리용)
            var go = new GameObject($"{name}_Storage");
            go.transform.SetParent(transform, false);
            storage = go.transform;
        }

        // 미리 몇 개 만들어서 큐에 넣어두기
        for (int i = 0; i < prewarm; i++)
        {
            var inst = CreateOne();
            Return(inst);
        }
    }

    GameObject CreateOne()
    {
        var go = Instantiate(prefab, storage);
        go.name = prefab.name;         // 보기 좋게 이름 정리
        go.SetActive(false);           // 기본은 비활성 상태
        return go;
    }

    public GameObject Get()
    {
        var go = (q.Count > 0) ? q.Dequeue() : CreateOne();

        if (!go)
            return null;

        go.transform.SetParent(null, true);
        go.SetActive(true);

        return go;
    }

    public void Return(GameObject go)
    {
        if (!go) return;

        go.SetActive(false);

        if (storage)
            go.transform.SetParent(storage, false);

        q.Enqueue(go);
    }
}
