visual style of The Kiseki Log

```toml
[brand]
description = "Overarching brand characteristics for The Kiseki Log teaser images."
backgrounds = "Clean, light, or off-white. Avoid heavily cluttered or dark-mode backgrounds."
preferred_colors = ["pastels", "soft gradients"]
avoid_colors = ["harsh neon", "highly saturated colors"]
simplicity = "Single, easily digestible main idea with plenty of negative space."

[styles]

  [styles.conceptual_abstract]
  name = "Conceptual & Abstract"
  use_cases = [
    "Essays",
    "High-level thoughts",
    "Roadmaps",
    "Philosophical reflections on AI"
  ]
  visual_elements = [
    "Surrealism",
    "Metaphorical imagery",
    "3D abstract shapes",
    "Soft gradients"
  ]
  composition = "Central focal point or clear directional flow."
  vibe = "Thought-provoking, futuristic but warm, and slightly dreamy."
  prompt_formula = """
A clean, modern, abstract digital illustration representing [Concept]. \
Use a mix of geometric and soft organic shapes. Color palette should \
feature soft pastel gradients [mention specific colors]. Minimalist \
composition, clean background, high conceptual art style, subtle 3D render feel.\
"""

  [styles.clean_technical]
  name = "Clean Technical"
  use_cases = [
    "Algorithms",
    "Tutorials",
    "Deep dives into models",
    "Math",
    "Architecture"
  ]
  visual_elements = [
    "Flowcharts",
    "Progressive sequences",
    "Clean vector lines",
    "Nodes",
    "Simple iconography"
  ]
  composition = "Highly structured, usually reading left-to-right or branching out logically."
  color_palette = "Stark white background with soft pastel colors (mint green, baby blue, soft lavender, peach) to differentiate nodes or steps."
  prompt_formula = """
A clean, minimalist educational diagram explaining [Process]. \
The image should have a pure white background. Use simple, flat vector \
graphics and circles connected by elegant lines. Color the elements using \
a soft pastel palette (mint green, light blue, pale pink). Academic, \
clear, highly structured, and visually soothing.\
"""
```
