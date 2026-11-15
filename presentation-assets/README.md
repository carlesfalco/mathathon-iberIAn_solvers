# SAIR presentation assets

Retrieved from the official [SAIR Foundation website](https://sair.foundation/) for the presentation restyle:

- `sair-mark.png`: official logo-only transparent mark from <https://sair.foundation/favicon-192x192.png>. The presentation places it at 3 mm height in the bottom-right corner, without the SAIR text.
- `SAIR-Heading.otf`: the heading font declared by the site stylesheet, **Helvetica Neue LT Pro 67 Medium Condensed**, from <https://sair.foundation/fonts/Helvetica%20Neue%20LT%20Pro%2067%20Medium%20Condensed.otf>.

The site uses **Helvetica Neue LT Pro 67 Medium Condensed** for titles and a Helvetica Neue / Inter / Segoe UI family for text. The deck uses the supplied font for titles, TeX Gyre Heros as the installed Helvetica-compatible text face, and native AMS/LaTeX mathematics.

- `caltech-sair-banner.png`: title-slide-only banner composed from [Caltech's official wordmark](https://www.caltech.edu/static/core/img/caltech-new-logo.png) and [SAIR's official full header SVG](https://sair.foundation/assets/header-logo-DPkSXSp3.svg). The SAIR header artwork is recoloured burgundy to match the supplied reference banner; its transparent background lets it sit cleanly on the title slide.

The visual theme uses the site's light-theme burgundy text (`#3B0E27`), copper hero accent (`#AB704B`), warm surface (`#F5F3F4`) and restrained related shades on a white background. Rebuild the deck with XeLaTeX (`latexmk -xelatex ...`).
