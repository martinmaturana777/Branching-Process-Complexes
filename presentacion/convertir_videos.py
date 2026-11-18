"""
Convierte los videos de ../videos en cuadros PNG nítidos para usarlos con
\\animategraphics (paquete animate) en presentacion_final.tex.

Cada video queda en  figuras/anim/<nombre>/  con:
  f-001.png ... f-NNN.png   cuadros a la resolución original del video (hasta ANCHO_MAX),
                            en PNG con paleta reducida: líneas y texto nítidos, sin el
                            desenfoque del JPEG;
  timeline.txt              orden de los cuadros para animate. Repite el último cuadro
                            PAUSA_FINAL segundos, para que se note la extinción antes de
                            volver a empezar. Repetir un cuadro no aumenta el peso del PDF:
                            animate reutiliza la misma imagen;
  param.tex                 número de cuadros y cuadros por segundo, que lee \\videogrande.

Si el video tiene más de N_MAX cuadros, se eligen N_MAX equiespaciados (time-lapse).
Siempre se incluyen el primer y el último cuadro, así se ve el estado final.

Uso (desde cualquier carpeta):
    python presentacion/convertir_videos.py            (todos)
    python presentacion/convertir_videos.py ninja_anillo   (solo los nombrados)
Requiere ffmpeg y ffprobe.
"""
import shutil
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
CARPETA_VIDEOS = AQUI.parent / "videos"
SALIDA = AQUI / "figuras" / "anim"
VIDEOS = [                      # los que usa la presentación
    "z2_10particulas_arbol",
    "z2_lazo_sin_arbol",
    "z2_T_creciente_1particula",
    "z2_T_creciente_lazo",
    "z3_10particulas_hasta_extincion",
    "ninja_anillo",
]
N_MAX = 120         # cuadros por animación como máximo
FPS_MIN = 10        # cuadros por segundo mínimos de la animación
PAUSA_FINAL = 1.0   # segundos que se mantiene el último cuadro
ANCHO_MAX = 1400    # ancho máximo en píxeles (si el video es más chico, no se agranda)
COLORES = 128       # colores de la paleta PNG por cuadro (64 = más liviano, 256 = máxima fidelidad)


def info_video(video: Path) -> tuple[int, float]:
    """(número real de cuadros, cuadros por segundo) de la pista de video."""
    salida = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=r_frame_rate,nb_read_packets", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True)
    tasa, total = salida.stdout.strip().split(",")[:2]
    num, den = tasa.split("/")
    return int(total), float(num) / float(den)


def indices_equiespaciados(total: int, n: int) -> list[int]:
    """n índices en [0, total-1], equiespaciados, que incluyen el primero y el último."""
    if n == 1:
        return [total - 1]
    return [round(i * (total - 1) / (n - 1)) for i in range(n)]


def extraer_todos(video: Path, destino: Path) -> list[Path]:
    """Extrae todos los cuadros del video como PNG con paleta propia por cuadro."""
    filtro = (f"[0:v:0]scale='min({ANCHO_MAX}\\,iw)':-2:flags=lanczos,"
              f"split[a][b];[a]palettegen=max_colors={COLORES}:stats_mode=single[p];"
              f"[b][p]paletteuse=new=1:dither=none[out]")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video),
         "-filter_complex", filtro, "-map", "[out]", "-fps_mode", "passthrough",
         str(destino / "tmp-%05d.png")],
        check=True)
    return sorted(destino.glob("tmp-*.png"))


def convertir(nombre: str) -> float:
    video = CARPETA_VIDEOS / f"{nombre}.mp4"
    destino = SALIDA / nombre
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)

    total, fps_video = info_video(video)
    tmp = extraer_todos(video, destino)
    n = min(len(tmp), N_MAX)
    elegidos = indices_equiespaciados(len(tmp), n)
    for i, j in enumerate(elegidos):
        tmp[j].rename(destino / f"f-{i + 1:03d}.png")
    for t in destino.glob("tmp-*.png"):
        t.unlink()

    # misma velocidad que el video si se usan todos sus cuadros; si no, al menos FPS_MIN
    fps = max(FPS_MIN, round(n * fps_video / len(tmp)))
    pausa = round(PAUSA_FINAL * fps)
    lineas = [f":{fps}:0"] + [f"::{i}" for i in range(1, n)] + [f"::{n - 1}"] * pausa
    (destino / "timeline.txt").write_text("\n".join(lineas) + "\n")
    (destino / "param.tex").write_text(
        f"\\def\\animultimo{{{n:03d}}}\\def\\animfps{{{fps}}}%\n")

    peso = sum(c.stat().st_size for c in destino.glob("f-*.png")) / 1e6
    print(f"{nombre:34s} {n:4d} de {total:4d} cuadros, {fps:2d} fps, "
          f"{n / fps:5.1f} s + {pausa / fps:.1f} s de pausa, {peso:5.1f} MB")
    return peso


def main() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        sys.exit("No encuentro ffmpeg. Instálalo y vuelve a abrir la terminal.")
    videos = sys.argv[1:] or VIDEOS
    faltan = [v for v in videos if not (CARPETA_VIDEOS / f"{v}.mp4").exists()]
    if faltan:
        sys.exit(f"Faltan videos en {CARPETA_VIDEOS}: {', '.join(faltan)}")
    peso = sum(convertir(v) for v in videos)
    print(f"\nListo: {peso:.1f} MB en {SALIDA}")


if __name__ == "__main__":
    main()
