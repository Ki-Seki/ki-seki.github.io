# MomentWriter Android App

A Jetpack Compose app for writing Hugo moments on mobile.

## Features

- Write moments with title, date, time, location, and mood
- Markdown editing with Hugo shortcode toolbar (admonition, details, github, media, bibtex)
- Add multiple images (automatically compressed and uploaded)
- Creates PRs directly via GitHub Git Data API
- All files uploaded in a single atomic commit

## Setup

1. Build the app:

   ```bash
   ./gradlew assembleDebug
   ```

2. Install the APK on your Android device (API 29+)

3. Configure GitHub settings:
   - GitHub Personal Access Token (PAT) with `repo` scope
   - Repository owner (e.g., `Ki-Seki`)
   - Repository name (e.g., `ki-seki.github.io`)
   - Default location

## Usage

1. Fill in the moment details (title, date, location, mood)
2. Write your content in Markdown
3. Use the shortcode toolbar to insert Hugo shortcodes
4. Add images using the + button
5. Tap the Send button to create a PR

## Permissions

- `INTERNET` - Required for GitHub API calls
- `READ_MEDIA_IMAGES` - Required for image selection (API 33+)
- `READ_EXTERNAL_STORAGE` - Required for image selection (API 32 and below)

## GitHub Actions

The repository includes a workflow that builds the debug APK on every push to the `android/` directory.

## Requirements

- Android API 29+ (Android 10+)
- Java 17
- Gradle 8.7
