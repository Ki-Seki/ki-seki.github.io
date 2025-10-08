# Workflow Documentation

## moment-from-issue.yml

This workflow automatically creates a moment post from a GitHub issue labeled with "moment".

### Timezone Configuration

The workflow uses a configurable timezone offset for the timestamp in moment posts. You can configure it in two ways:

#### Option 1: Repository Variable (Recommended)

1. Go to your repository's Settings → Secrets and variables → Actions → Variables
2. Create a new repository variable named `TIMEZONE_OFFSET`
3. Set the value to your timezone offset (e.g., `+08:00`, `-05:00`, `+00:00`)

This method allows you to change the timezone without modifying the workflow file.

#### Option 2: Edit Workflow File

Edit the `TIMEZONE_OFFSET` value in the workflow file directly:

```yaml
env:
  TIMEZONE_OFFSET: '+08:00'  # Change this to your timezone offset
```

### Timezone Offset Examples

- Beijing, Shanghai (China Standard Time): `+08:00`
- Tokyo (Japan Standard Time): `+09:00`
- New York (EST): `-05:00` or (EDT): `-04:00`
- London (GMT): `+00:00` or (BST): `+01:00`
- Los Angeles (PST): `-08:00` or (PDT): `-07:00`
- UTC: `+00:00`

### Default Behavior

If neither repository variable nor workflow file value is set, the default timezone offset is `+08:00` (China Standard Time).
