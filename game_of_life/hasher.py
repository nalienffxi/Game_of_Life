import hashlib

def compute_hash(quadtree):
    serialized_quadtree = quadtree.serialize()
    hasher = hashlib.sha256()
    hasher.update(serialized_quadtree.encode('utf-8'))
    return hasher.hexdigest()

