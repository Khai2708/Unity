using UnityEngine;

public class Node
{
    public bool walkable;
    public Vector3 worldPosition;
    public int gridX;
    public int gridY;

    public int gCost;
    public int hCost;
    public Node parent;

    public int fCost => gCost + hCost;

    public Node(bool walkable, Vector3 worldPos, int gridX, int gridY)
    {
        this.walkable = walkable;
        this.worldPosition = worldPos;
        this.gridX = gridX;
        this.gridY = gridY;
    }
}
