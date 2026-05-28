package com.kiseki.momentwriter

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Code
import androidx.compose.material.icons.outlined.ExpandMore
import androidx.compose.material.icons.outlined.FormatQuote
import androidx.compose.material.icons.outlined.Image
import androidx.compose.material.icons.outlined.Info
import androidx.compose.ui.graphics.vector.ImageVector
import java.security.MessageDigest
import java.time.LocalDate
import java.time.format.DateTimeFormatter

// --- Slug & Filename Generation ---

fun titleToSlug(title: String): String =
    title.lowercase()
        .replace(Regex("[^\\w\\s-]"), "")
        .replace(Regex("\\s+"), "-")
        .replace(Regex("-+"), "-")
        .trim()
        .let { if (it.length > 50) it.take(50).trimEnd('-') else it }

fun formatDatePrefix(date: LocalDate): String =
    date.format(DateTimeFormatter.ofPattern("yyMMdd"))

fun buildDirName(date: LocalDate, title: String): String =
    "${formatDatePrefix(date)}-${titleToSlug(title)}"

fun makeImageFilename(originalName: String, bytes: ByteArray): String {
    val hash = MessageDigest.getInstance("SHA-1").digest(bytes)
        .take(4).joinToString("") { "%02x".format(it) }
    val clean = originalName.replace(Regex("[^\\w.\\-]+"), "-").take(50)
    return "$hash-$clean"
}

// --- Shortcode Templates ---

data class ShortcodeTemplate(
    val label: String,
    val icon: ImageVector,
    val template: String,
    val cursorOffset: Int
)

val ADMONITION_TYPES = listOf(
    "note", "abstract", "info", "tip", "success", "question",
    "warning", "failure", "danger", "bug", "example", "quote"
)

fun makeAdmonition(type: String) = ShortcodeTemplate(
    label = type.replaceFirstChar { it.uppercase() },
    icon = Icons.Outlined.Info,
    template = """{{< admonition type=$type title="Title" open=false >}}
content
{{< /admonition >}}""",
    cursorOffset = 30
)

val SHORTCODE_TEMPLATES = listOf(
    ShortcodeTemplate(
        "Media", Icons.Outlined.Image,
        """{{< media
src="file.jpg"
caption=""
>}}""", 16
    )
) + ADMONITION_TYPES.map { makeAdmonition(it) } + listOf(
    ShortcodeTemplate(
        "Details", Icons.Outlined.ExpandMore,
        """{{< details "Title" >}}
content
{{< /details >}}""", 15
    ),
    ShortcodeTemplate(
        "GitHub", Icons.Outlined.Code,
        """{{< github "user/repo" >}}""", 14
    ),
    ShortcodeTemplate(
        "BibTeX", Icons.Outlined.FormatQuote,
        "{{< bibtex >}}", -1
    )
)

val MOOD_PRESETS = listOf(
    "Happy", "Calm", "Peace", "Touching", "Grateful", "Excited",
    "Nostalgia", "Cozy", "Sad", "Anxious", "Moved", "Proud", "Rainy", "Thinking"
)
