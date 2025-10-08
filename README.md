# The Kiseki Log

If you find this blog helpful, please consider giving it a ⭐ **star** or 👀 **watching** the repository to get notified about the latest articles and updates.

Also feel free to open ⁉️ **issues** or 🤝 **pull requests** for suggestions and improvements!

## Deployment

To serve the site locally:

1. **Install [Hugo](https://gohugo.io/getting-started/installing/)** (extended version recommended).
2. Clone this repository:

   ```zsh
   git clone https://github.com/ki-seki/ki-seki.github.io.git
   cd ki-seki.github.io
   ```

3. Start the local server:

   ```zsh
   hugo server -D
   ```

   - The site will be available at [http://localhost:1313](http://localhost:1313).
   - The `-D` flag includes draft posts.

## Creating Moments

You can create a new moment in two ways:

### Method 1: Using GitHub Issues (Recommended)

1. Go to the [Issues tab](https://github.com/ki-seki/ki-seki.github.io/issues/new/choose)
2. Select **"Create a Moment"** template
3. Fill in the details:
   - **Title**: The title of your moment
   - **Content**: Your moment content (supports Markdown and Hugo shortcodes)
   - **Location**: Optional location tag
   - **Mood**: Optional mood tag
   - **Draft Status**: Whether to publish as draft
4. Submit the issue with the `moment` label
5. A pull request will be automatically created with the new moment
6. Review and merge the PR to publish the moment

### Method 2: Manual Creation

Use Hugo's archetype command:

```zsh
hugo new content/moments/YYMMDD-title/index.md
```

Then edit the created file to add your content.

## License

<a href="https://ki-seki.github.io/">The Kiseki Log</a> © 2023-2025 by <a href="https://ki-seki.github.io/cv/">Shichao Song</a> is licensed under <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a><span style="margin-left:.2em;"></span>
<img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" alt="Creative Commons" style="height:1em;"><span style="margin-left:.2em;"></span>
<img src="https://mirrors.creativecommons.org/presskit/icons/by.svg" alt="Attribution" style="height:1em;"><span style="margin-left:.2em;"></span>
<img src="https://mirrors.creativecommons.org/presskit/icons/sa.svg" alt="ShareAlike" style="height:1em;">
