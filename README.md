This repository contains the codes to generate the videos of my YouTube channel [Varniex](https://youtube.com/@Varniex)

> [!Note]
> The videos are animated using Grant Sanderson's ([3Blue1Brown](https://www.3blue1brown.com/)) library [ManimGL](https://github.com/3b1b/manim) `v 1.7.2`.
> This repository works with [this](https://github.com/3b1b/manim/tree/7a7bf83f117034b5cdf60ae85511c1b004769651) commit of **ManimGL**.

## Manim Versions
There are actually three versions of Manim:
1. Manim OpenGL ([ManimGL](https://github.com/3b1b/manim)): This version is maintained by 3Blue1Brown.
2. [ManimCE](https://manim.community) ([GitHub repo](https://github.com/ManimCommunity/manim)): This version is community maintained. It is better documented for beginners especially. If you are a newbie, I'd suggest you to go with this.
3. [Manim Cairo](https://github.com/3b1b/manim/tree/cairo-backend): This is **deprecated** now.

> [!Warning]
> All of these versions are very different from each other. Please follow the guidelines of the respective version of Manim for its installation and tutorials.

### Change the $LaTeX$ font to "Cambria"
Before rendering the video:

* Change the default template in the `custom_config.yml` to `ctex` like:
```yml
tex:
  template: "ctex"
```

* Change the main (and math) font to "Cambria" of `ctex` preamble in the `tex_templates.yml` file in the "manimlib" folder (ManimGL).
```yml
\usepackage{unicode-math}
\setmainfont{Cambria}
\setmathfont{Cambria Math}
```

### License

The library ManimGL itself is open source and under MIT License.

But, the contents of this repository are for references only and licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/), and only to be used to generate the videos for [Varniex](https://youtube.com/@Varniex) YouTube Channel.

Copyright &copy; 2023 - 2025 Varniex
