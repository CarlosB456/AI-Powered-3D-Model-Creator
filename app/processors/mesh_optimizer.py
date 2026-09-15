"""
AI-Powered 3D Model Creator - Mesh Optimizer Module
Mesh decimation (polygon reduction), cleanup, recentering, and scaling.
"""
import os
import trimesh
import numpy as np

def decimate_mesh(mesh: trimesh.Trimesh, target_faces: int = 50000) -> trimesh.Trimesh:
    """
    Reduces polygon count using Quadric Edge Collapse Decimation (PyMeshLab).
    Preserves UV coordinates and geometry boundary shapes.
    """
    current_faces = len(mesh.faces)
    if target_faces <= 0 or target_faces >= current_faces:
        return mesh

    print(f"[OPTIMIZER] Reduciendo polígonos de {current_faces:,} a {target_faces:,} caras...", flush=True)
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        m = pymeshlab.Mesh(
            vertex_matrix=mesh.vertices.astype(np.float64),
            face_matrix=mesh.faces.astype(np.int32)
        )
        ms.add_mesh(m)
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(target_faces),
            qualitythr=0.3,
            preserveboundary=True,
            preservenormal=True,
            preservetopology=True
        )
        dec_mesh = ms.current_mesh()
        new_vertices = dec_mesh.vertex_matrix().astype(np.float32)
        new_faces = dec_mesh.face_matrix().astype(np.int32)
        
        result = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=True)
        print(f"[OPTIMIZER] Malla optimizada: {len(result.faces):,} caras.", flush=True)
        return result
    except Exception as e:
        print(f"[WARN] Error durante decimation ({e}), manteniendo malla original.", flush=True)
        return mesh

def normalize_mesh(mesh: trimesh.Trimesh, target_scale: float = 1.0) -> trimesh.Trimesh:
    """
    Centers the mesh at origin (0, 0, 0) and normalizes bounding box size.
    """
    try:
        # Center mesh
        bbox_min = mesh.vertices.min(axis=0)
        bbox_max = mesh.vertices.max(axis=0)
        center = (bbox_min + bbox_max) / 2.0
        mesh.vertices -= center

        # Normalize scale
        extent = (bbox_max - bbox_min).max()
        if extent > 0:
            mesh.vertices *= (target_scale / extent)
        return mesh
    except Exception as e:
        print(f"[WARN] Normalization failed: {e}")
        return mesh

def export_mesh_files(mesh: trimesh.Trimesh, output_dir: str, base_name: str) -> tuple[str, str]:
    """
    Exports mesh to both GLB and OBJ formats.
    Returns (glb_path, obj_path).
    """
    os.makedirs(output_dir, exist_ok=True)
    glb_path = os.path.join(output_dir, f"{base_name}.glb")
    obj_path = os.path.join(output_dir, f"{base_name}.obj")
    
    mesh.export(glb_path)
    mesh.export(obj_path)
    
    return glb_path, obj_path
