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

## Moment Posts

This repository includes an automated workflow to create moment posts from GitHub issues. 

### Timezone Configuration

By default, moment posts use `+08:00` (China Standard Time) timezone. To change the timezone:

1. Go to Settings → Secrets and variables → Actions → Variables
2. Create a variable named `TIMEZONE_OFFSET` with your timezone offset (e.g., `-05:00`, `+09:00`)

See [.github/workflows/README.md](.github/workflows/README.md) for more details.

## License

<a href="https://ki-seki.github.io/">The Kiseki Log</a> © 2023-2025 by <a href="https://ki-seki.github.io/cv/">Shichao Song</a> is licensed under <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a><span style="margin-left:.2em;"></span>
<img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" alt="Creative Commons" style="height:1em;"><span style="margin-left:.2em;"></span>
<img src="https://mirrors.creativecommons.org/presskit/icons/by.svg" alt="Attribution" style="height:1em;"><span style="margin-left:.2em;"></span>
<img src="https://mirrors.creativecommons.org/presskit/icons/sa.svg" alt="ShareAlike" style="height:1em;">
