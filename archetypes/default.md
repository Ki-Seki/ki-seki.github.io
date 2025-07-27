---
date: '{{ .Date }}'
title: '{{ replace .File.ContentBaseName "-" " " | title }}'
summary: ''
tags: []
math: true
---

## Common Elements

### Blockquote with attribution

> Don't communicate by sharing memory, share memory by communicating.
>
> — <cite>Rob Pike[^1]</cite>

[^1]: The above quote is excerpted from Rob Pike's [talk](https://www.youtube.com/watch?v=PAAkCSZUG1c) during Gopherfest, November 18, 2015.

### Math

You can use inline math like this: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$.

Or:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

### Code block with line numbers and <mark>highlighted</mark> lines

```html {linenos=true,hl_lines=[2,8]}
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Example HTML5 Document</title>
    <meta
      name="description"
      content="Sample article showcasing basic Markdown syntax and formatting for HTML elements."
    />
  </head>
  <body>
    <p>Test</p>
  </body>
</html>
```

### Other Elements — abbr, sub, sup, kbd, mark

<abbr title="Graphics Interchange Format">GIF</abbr> is a bitmap image format.

H<sub>2</sub>O

X<sup>n</sup> + Y<sup>n</sup> = Z<sup>n</sup>

Press <kbd><kbd>CTRL</kbd>+<kbd>ALT</kbd>+<kbd>Delete</kbd></kbd> to end the session.

Most <mark>salamanders</mark> are nocturnal, and hunt for insects, worms, and other small creatures.

## Shortcodes

### Github Gist

{{< gist adityatelange 376cd56ee2c94aaa2e8b93200f2ba8b5 >}}

### Figure Shortcode

{{< figure src="https://images.unsplash.com/photo-1702382930514-9759f4ca5469" attr="Photo by [Aditya Telange](https://unsplash.com/@adityatelange?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash) on [Unsplash](https://unsplash.com/photos/Z0lL0okYjy0?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash)" align=center link="https://unsplash.com/photos/a-person-sitting-on-a-rock-wall-looking-out-at-the-ocean-Z0lL0okYjy0" target="_blank" >}}

### YouTube

{{< youtube hjD9jTi_DQ4 >}}

### X (Twitter) Shortcode

{{< x user="adityatelange" id="1724414854348357922" >}}

### Vimeo Shortcode

{{< vimeo 152985022 >}}

### Details Shortcode

{{< details "Click me">}}
This is a collapsible section. You can put any content here, including text, images, or even other shortcodes. Click the summary to expand or collapse this section.
{{< /details >}}

## Citation

```bibtex

```

## References

## Appendix

{{<details>}}

{{</details>}}
