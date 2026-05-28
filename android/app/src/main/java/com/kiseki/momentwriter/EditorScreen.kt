package com.kiseki.momentwriter

import android.content.Context
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarDuration
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.SnackbarResult
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import coil.compose.AsyncImage
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import java.time.LocalDate
import java.time.LocalTime
import java.time.format.DateTimeFormatter

// --- ViewModel ---

class EditorViewModel(private val settingsRepo: SettingsRepository) : ViewModel() {
    var title by mutableStateOf("")
    var date by mutableStateOf(LocalDate.now())
    var time by mutableStateOf(LocalTime.now())
    var location by mutableStateOf("")
    var mood by mutableStateOf("")
    var contentField by mutableStateOf(TextFieldValue(""))
    var images by mutableStateOf<List<DraftImage>>(emptyList())
    var uploadState by mutableStateOf<UploadState>(UploadState.Idle)

    init {
        viewModelScope.launch {
            settingsRepo.settingsFlow.first().let { location = it.defaultLocation }
        }
    }

    fun insertShortcode(template: ShortcodeTemplate) {
        val text = contentField.text
        val pos = contentField.selection.start
        val newText = text.substring(0, pos) + template.template + text.substring(pos)
        val cursor = if (template.cursorOffset >= 0) pos + template.cursorOffset
        else pos + template.template.length
        contentField = TextFieldValue(newText, androidx.compose.ui.text.TextRange(cursor))
    }

    fun addImage(context: Context, uri: Uri, originalName: String) {
        viewModelScope.launch {
            try {
                val bytes = GitHubApi.compressImage(context, uri)
                val filename = makeImageFilename(originalName, bytes)
                images = images + DraftImage(filename, bytes)
            } catch (e: Exception) {
                uploadState = UploadState.Error("Failed to add image: ${e.message}")
            }
        }
    }

    fun removeImage(index: Int) {
        images = images.toMutableList().apply { removeAt(index) }
    }

    fun submit() {
        viewModelScope.launch {
            val settings = settingsRepo.settingsFlow.first()
            if (!settings.isConfigured) {
                uploadState = UploadState.Error("Please configure GitHub settings first")
                return@launch
            }
            if (title.isBlank()) {
                uploadState = UploadState.Error("Title cannot be empty")
                return@launch
            }

            uploadState = UploadState.Uploading("Creating PR...")
            try {
                val draft = MomentDraft(
                    title = title,
                    date = java.time.LocalDateTime.of(date, time),
                    location = location,
                    mood = mood,
                    content = contentField.text,
                    images = images
                )
                val api = GitHubApi(settings.owner, settings.repo, settings.token)
                val result = api.uploadMoment(draft)
                uploadState = UploadState.Success(result.htmlUrl)
            } catch (e: Exception) {
                uploadState = UploadState.Error(e.message ?: "Upload failed")
            }
        }
    }
}

class EditorVMFactory(private val repo: SettingsRepository) : ViewModelProvider.Factory {
    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T = EditorViewModel(repo) as T
}

// --- Screen ---

@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun EditorScreen(onNavigateToSettings: () -> Unit, settingsRepo: SettingsRepository) {
    val viewModel: EditorViewModel = viewModel(factory = EditorVMFactory(settingsRepo))
    val context = LocalContext.current
    val snackbarHostState = remember { SnackbarHostState() }
    val scope = rememberCoroutineScope()

    val imagePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickMultipleVisualMedia()
    ) { uris ->
        uris.forEachIndexed { _, uri ->
            val name = uri.lastPathSegment ?: "image.jpg"
            viewModel.addImage(context, uri, name)
        }
    }

    // Snackbar for upload state
    LaunchedEffect(viewModel.uploadState) {
        when (val state = viewModel.uploadState) {
            is UploadState.Success -> {
                val result = snackbarHostState.showSnackbar(
                    message = "PR created!",
                    actionLabel = "Open",
                    duration = SnackbarDuration.Long
                )
                if (result == SnackbarResult.ActionPerformed) {
                    // Open URL - user can implement if needed
                }
                viewModel.uploadState = UploadState.Idle
            }
            is UploadState.Error -> {
                snackbarHostState.showSnackbar(state.message, duration = SnackbarDuration.Long)
                viewModel.uploadState = UploadState.Idle
            }
            else -> {}
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("New Moment") },
                actions = {
                    IconButton(onClick = onNavigateToSettings) {
                        Icon(Icons.Default.Settings, "Settings")
                    }
                    IconButton(
                        onClick = { viewModel.submit() },
                        enabled = viewModel.uploadState !is UploadState.Uploading
                    ) {
                        Icon(Icons.Default.Send, "Submit")
                    }
                }
            )
        },
        snackbarHost = { SnackbarHost(snackbarHostState) }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(horizontal = 16.dp)
        ) {
            Column(
                modifier = Modifier
                    .weight(1f)
                    .verticalScroll(rememberScrollState())
            ) {
                // Title
                OutlinedTextField(
                    value = viewModel.title,
                    onValueChange = { viewModel.title = it },
                    placeholder = { Text("Title") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )

                Spacer(Modifier.height(8.dp))

                // Date & Time chips
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    FilterChip(
                        selected = false,
                        onClick = { /* DatePickerDialog - simplified for now */ },
                        label = { Text(viewModel.date.format(DateTimeFormatter.ofPattern("MMM d, yyyy"))) }
                    )
                    FilterChip(
                        selected = false,
                        onClick = { /* TimePickerDialog - simplified for now */ },
                        label = { Text(viewModel.time.format(DateTimeFormatter.ofPattern("h:mm a"))) }
                    )
                }

                Spacer(Modifier.height(8.dp))

                // Location & Mood
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    OutlinedTextField(
                        value = viewModel.location,
                        onValueChange = { viewModel.location = it },
                        placeholder = { Text("Location") },
                        singleLine = true,
                        modifier = Modifier.weight(1f)
                    )
                    MoodField(
                        value = viewModel.mood,
                        onValueChange = { viewModel.mood = it },
                        modifier = Modifier.weight(1f)
                    )
                }

                Spacer(Modifier.height(8.dp))

                // Image strip
                LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    items(viewModel.images.size) { index ->
                        val img = viewModel.images[index]
                        Box(modifier = Modifier.size(100.dp)) {
                            AsyncImage(
                                model = img.bytes,
                                contentDescription = null,
                                modifier = Modifier
                                    .fillMaxSize()
                                    .clip(RoundedCornerShape(8.dp))
                                    .border(1.dp, Color.Gray, RoundedCornerShape(8.dp)),
                                contentScale = ContentScale.Crop
                            )
                            IconButton(
                                onClick = { viewModel.removeImage(index) },
                                modifier = Modifier
                                    .align(Alignment.TopEnd)
                                    .size(24.dp)
                                    .background(Color.Black.copy(alpha = 0.5f), RoundedCornerShape(4.dp))
                            ) {
                                Icon(Icons.Default.Close, "Remove", tint = Color.White, modifier = Modifier.size(16.dp))
                            }
                        }
                    }
                    item {
                        Box(
                            modifier = Modifier
                                .size(100.dp)
                                .clip(RoundedCornerShape(8.dp))
                                .border(1.dp, Color.Gray, RoundedCornerShape(8.dp))
                                .clickable {
                                    imagePicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                                },
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(Icons.Default.Add, "Add image", modifier = Modifier.size(32.dp))
                        }
                    }
                }

                Spacer(Modifier.height(8.dp))
            }

            // Upload progress
            if (viewModel.uploadState is UploadState.Uploading) {
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                Spacer(Modifier.height(4.dp))
            }

            // Content editor
            BasicTextField(
                value = viewModel.contentField,
                onValueChange = { viewModel.contentField = it },
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(2f)
                    .padding(vertical = 8.dp),
                textStyle = TextStyle(fontSize = 14.sp, color = MaterialTheme.colorScheme.onSurface),
                cursorBrush = SolidColor(MaterialTheme.colorScheme.primary),
                decorationBox = { innerTextField ->
                    Box {
                        if (viewModel.contentField.text.isEmpty()) {
                            Text("Write your moment...", color = Color.Gray, fontSize = 14.sp)
                        }
                        innerTextField()
                    }
                }
            )

            // Shortcode toolbar
            ShortcodeToolbar(onInsert = { viewModel.insertShortcode(it) })
        }
    }
}

@Composable
private fun MoodField(value: String, onValueChange: (String) -> Unit, modifier: Modifier) {
    var expanded by remember { mutableStateOf(false) }

    Box(modifier = modifier) {
        OutlinedTextField(
            value = value,
            onValueChange = onValueChange,
            placeholder = { Text("Mood") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth()
        )
        DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
            MOOD_PRESETS.forEach { preset ->
                DropdownMenuItem(
                    text = { Text(preset) },
                    onClick = { onValueChange(preset); expanded = false }
                )
            }
        }
    }
}

@Composable
private fun ShortcodeToolbar(onInsert: (ShortcodeTemplate) -> Unit) {
    var showAdmonitionMenu by remember { mutableStateOf(false) }

    // Get non-admonition templates
    val mediaTemplate = SHORTCODE_TEMPLATES.first { it.label == "Media" }
    val detailsTemplate = SHORTCODE_TEMPLATES.first { it.label == "Details" }
    val githubTemplate = SHORTCODE_TEMPLATES.first { it.label == "GitHub" }
    val bibtexTemplate = SHORTCODE_TEMPLATES.first { it.label == "BibTeX" }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceEvenly
    ) {
        TextButton(onClick = { onInsert(mediaTemplate) }) { Text("Media", fontSize = 11.sp) }

        Box {
            TextButton(onClick = { showAdmonitionMenu = true }) { Text("Admonition", fontSize = 11.sp) }
            DropdownMenu(expanded = showAdmonitionMenu, onDismissRequest = { showAdmonitionMenu = false }) {
                ADMONITION_TYPES.forEach { type ->
                    DropdownMenuItem(
                        text = { Text(type) },
                        onClick = {
                            onInsert(makeAdmonition(type))
                            showAdmonitionMenu = false
                        }
                    )
                }
            }
        }

        TextButton(onClick = { onInsert(detailsTemplate) }) { Text("Details", fontSize = 11.sp) }
        TextButton(onClick = { onInsert(githubTemplate) }) { Text("GitHub", fontSize = 11.sp) }
        TextButton(onClick = { onInsert(bibtexTemplate) }) { Text("BibTeX", fontSize = 11.sp) }
    }
}
