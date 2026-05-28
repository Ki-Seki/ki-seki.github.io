package com.kiseki.momentwriter

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Base64
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.time.LocalDateTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

class GitHubApi(private val owner: String, private val repo: String, private val token: String) {

    private val client = OkHttpClient()
    private val json = Json { ignoreUnknownKeys = true }
    private val base = "https://api.github.com/repos/$owner/$repo"
    private val jsonMedia = "application/json; charset=utf-8".toMediaType()

    private fun Request.Builder.githubHeaders() = this
        .addHeader("Authorization", "Bearer $token")
        .addHeader("Accept", "application/vnd.github+json")
        .addHeader("X-GitHub-Api-Version", "2022-11-28")

    private suspend fun Call.await(): Response = suspendCancellableCoroutine { cont ->
        cont.invokeOnCancellation { cancel() }
        enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                if (cont.isActive) cont.resumeWithException(e)
            }
            override fun onResponse(call: Call, response: Response) {
                if (cont.isActive) cont.resume(response)
            }
        })
    }

    private inline fun <reified T> Request.Builder.jsonBody(body: T): Request.Builder {
        val jsonString = json.encodeToString(body)
        return post(jsonString.toRequestBody(jsonMedia))
    }

    private fun checkResponse(response: Response, action: String) {
        if (!response.isSuccessful) {
            val code = response.code
            val body = response.body?.string() ?: ""
            throw IOException("GitHub API $action failed ($code): $body")
        }
    }

    // --- API Methods ---

    suspend fun getDefaultBranch(): String {
        val request = Request.Builder()
            .url("https://api.github.com/repos/$owner/$repo")
            .githubHeaders()
            .get()
            .build()
        val response = client.newCall(request).await()
        checkResponse(response, "getRepo")
        val body = response.body?.string() ?: throw IOException("Empty response")
        return json.decodeFromString<RepoInfo>(body).defaultBranch
    }

    suspend fun getHeadSha(branch: String): String {
        val request = Request.Builder()
            .url("$base/git/ref/heads/$branch")
            .githubHeaders()
            .get()
            .build()
        val response = client.newCall(request).await()
        checkResponse(response, "getHeadSha")
        val body = response.body?.string() ?: throw IOException("Empty response")
        return json.decodeFromString<GitRefResp>(body).obj.sha
    }

    suspend fun createBranch(branchName: String, baseSha: String) {
        val request = Request.Builder()
            .url("$base/git/refs")
            .githubHeaders()
            .jsonBody(GitRefReq("refs/heads/$branchName", baseSha))
            .build()
        val response = client.newCall(request).await()
        checkResponse(response, "createBranch")
    }

    suspend fun createBlob(bytes: ByteArray): String {
        val content = Base64.encodeToString(bytes, Base64.NO_WRAP)
        val request = Request.Builder()
            .url("$base/git/blobs")
            .githubHeaders()
            .jsonBody(GitBlobReq(content, "base64"))
            .build()
        val response = client.newCall(request).await()
        checkResponse(response, "createBlob")
        val body = response.body?.string() ?: throw IOException("Empty response")
        return json.decodeFromString<GitBlobResp>(body).sha
    }

    suspend fun createTree(baseSha: String, entries: List<TreeEntry>): String {
        val request = Request.Builder()
            .url("$base/git/trees")
            .githubHeaders()
            .jsonBody(GitTreeReq(baseSha, entries))
            .build()
        val response = client.newCall(request).await()
        checkResponse(response, "createTree")
        val body = response.body?.string() ?: throw IOException("Empty response")
        return json.decodeFromString<GitTreeResp>(body).sha
    }

    suspend fun createCommit(message: String, treeSha: String, parentSha: String): String {
        val request = Request.Builder()
            .url("$base/git/commits")
            .githubHeaders()
            .jsonBody(GitCommitReq(message, treeSha, listOf(parentSha)))
            .build()
        val response = client.newCall(request).await()
        checkResponse(response, "createCommit")
        val body = response.body?.string() ?: throw IOException("Empty response")
        return json.decodeFromString<GitCommitResp>(body).sha
    }

    suspend fun updateRef(branch: String, sha: String) {
        val request = Request.Builder()
            .url("$base/git/refs/heads/$branch")
            .githubHeaders()
            .patch(json.encodeToString(GitRefUpdateReq(sha)).toRequestBody(jsonMedia))
            .build()
        val response = client.newCall(request).await()
        checkResponse(response, "updateRef")
    }

    suspend fun createPullRequest(title: String, body: String, head: String, baseBranch: String): PRResp {
        val request = Request.Builder()
            .url("$base/pulls")
            .githubHeaders()
            .jsonBody(PRReq(title, body, head, baseBranch))
            .build()
        val response = client.newCall(request).await()
        checkResponse(response, "createPR")
        val respBody = response.body?.string() ?: throw IOException("Empty response")
        return json.decodeFromString<PRResp>(respBody)
    }

    // --- Orchestration ---

    suspend fun uploadMoment(draft: MomentDraft): PRResp {
        val baseBranch = getDefaultBranch()
        val baseSha = getHeadSha(baseBranch)
        val dirName = buildDirName(draft.date.toLocalDate(), draft.title)
        // Add timestamp to ensure unique branch names
        val timestamp = System.currentTimeMillis()
        val branchName = "moment/$dirName-$timestamp"

        createBranch(branchName, baseSha)

        val indexMd = buildIndexMd(draft)
        val indexBlobSha = createBlob(indexMd.toByteArray(Charsets.UTF_8))

        val allEntries = mutableListOf(
            TreeEntry("content/moments/$dirName/index.md", sha = indexBlobSha)
        )
        for (img in draft.images) {
            val blobSha = createBlob(img.bytes)
            allEntries.add(TreeEntry("content/moments/$dirName/${img.filename}", sha = blobSha))
        }

        val treeSha = createTree(baseSha, allEntries)
        val commitSha = createCommit("post: add moment, $dirName", treeSha, baseSha)
        updateRef(branchName, commitSha)

        return createPullRequest(
            title = "post: add moment, $dirName",
            body = "New moment: ${draft.title}\n\nLocation: ${draft.location}\nMood: ${draft.mood}",
            head = branchName,
            baseBranch = baseBranch
        )
    }

    private fun buildIndexMd(draft: MomentDraft): String = buildString {
        appendLine("---")
        appendLine("title: \"${escapeYamlString(draft.title)}\"")
        appendLine("date: ${formatDateWithOffset(draft.date)}")
        appendLine("draft: false")
        appendLine("location: \"${escapeYamlString(draft.location)}\"")
        appendLine("mood: \"${escapeYamlString(draft.mood)}\"")
        appendLine("---")
        appendLine()
        append(draft.content)
    }

    private fun escapeYamlString(s: String): String =
        s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n")

    private fun formatDateWithOffset(dt: LocalDateTime): String {
        val zoned = dt.atZone(ZoneId.systemDefault())
        val offset = zoned.offset
        return "${dt.format(DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss"))}$offset"
    }

    companion object {
        fun compressImage(context: Context, uri: Uri, maxSize: Int = 1920): ByteArray {
            val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            context.contentResolver.openInputStream(uri)?.use {
                BitmapFactory.decodeStream(it, null, opts)
            }

            val width = opts.outWidth
            val height = opts.outHeight
            var sampleSize = 1
            while (width / sampleSize > maxSize || height / sampleSize > maxSize) {
                sampleSize *= 2
            }

            val decodeOpts = BitmapFactory.Options().apply { inSampleSize = sampleSize }
            val bitmap = context.contentResolver.openInputStream(uri)?.use {
                BitmapFactory.decodeStream(it, null, decodeOpts)
            } ?: throw IOException("Failed to decode image")

            val scaled = if (bitmap.width > maxSize || bitmap.height > maxSize) {
                val ratio = maxSize.toFloat() / maxOf(bitmap.width, bitmap.height)
                val newW = (bitmap.width * ratio).toInt()
                val newH = (bitmap.height * ratio).toInt()
                Bitmap.createScaledBitmap(bitmap, newW, newH, true).also {
                    if (it !== bitmap) bitmap.recycle()
                }
            } else {
                bitmap
            }

            val output = ByteArrayOutputStream()
            scaled.compress(Bitmap.CompressFormat.JPEG, 85, output)
            scaled.recycle()
            return output.toByteArray()
        }
    }
}
