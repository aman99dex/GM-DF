from pathlib import Path
r = len(list(Path('datasets/FaceForensics/real').rglob('*.jpg')))
f = len(list(Path('datasets/FaceForensics/fake').rglob('*.jpg')))
print(f"Real: {r}, Fake: {f}, Ratio: {f/r if r > 0 else 0:.2f}")
