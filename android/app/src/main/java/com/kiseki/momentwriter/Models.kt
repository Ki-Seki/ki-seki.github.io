package com.kiseki.momentwriter

import android.net.Uri
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import java.time.LocalDateTime

// --- App Settings ---

data class AppSettings(
    val token: String = "",
    val owner: String = "Ki-Seki",
    val repo: String = "ki-seki.github.io",
    val defaultLocation: String = "Beijing"
) {
    val isConfigured get() = token.isNotBlank() && owner.isNotBlank() && repo.isNotBlank()
}

// --- Moment Draft ---

data class MomentDraft(
    val title: String,
    val date: LocalDateTime,
    val location: String,
    val mood: String,
    val content: String,
    val images: List<DraftImage>
)

data class DraftImage(
    val filename: String,
    val bytes: ByteArray
)

// --- Upload State ---

sealed class UploadState {
    data object Idle : UploadState()
    data class Uploading(val message: String) : UploadState()
    data class Success(val prUrl: String) : UploadState()
    data class Error(val message: String) : UploadState()
}

// --- GitHub API DTOs ---

@Serializable
data class RepoInfo(@SerialName("default_branch") val defaultBranch: String)

@Serializable
data class GitRefResp(val ref: String, @SerialName("object") val obj: GitRefObj)

@Serializable
data class GitRefObj(val sha: String)

@Serializable
data class GitRefReq(val ref: String, val sha: String)

@Serializable
data class GitRefUpdateReq(val sha: String, val force: Boolean = false)

@Serializable
data class GitBlobReq(val content: String, val encoding: String)

@Serializable
data class GitBlobResp(val sha: String)

@Serializable
data class TreeEntry(
    val path: String,
    val mode: String = "100644",
    val type: String = "blob",
    val sha: String
)

@Serializable
data class GitTreeReq(@SerialName("base_tree") val baseTree: String, val tree: List<TreeEntry>)

@Serializable
data class GitTreeResp(val sha: String)

@Serializable
data class GitCommitReq(val message: String, val tree: String, val parents: List<String>)

@Serializable
data class GitCommitResp(val sha: String)

@Serializable
data class PRReq(val title: String, val body: String, val head: String, val base: String)

@Serializable
data class PRResp(val number: Int, @SerialName("html_url") val htmlUrl: String)
