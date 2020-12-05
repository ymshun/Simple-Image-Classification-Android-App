package com.example.imageclassificationapp

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.provider.MediaStore
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.imageclassificationapp.ml.MobilenetV2
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.support.image.TensorImage

class MainActivity : AppCompatActivity() {

    companion object {
        private const val GALLERY_REQUEST_CODE = 100
    }

    private var bitmap: Bitmap? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // RUN ボタンクリックで推論結果を表示
        inferrenceBtn.setOnClickListener {
            if (bitmap == null) return@setOnClickListener
            lifecycleScope.launch {
                // 推論結果を取得
                val outputs = classifyImage(bitmap!!)
                // viewにフォーマットした推論結果を15個表示
                resultTextView.text = outputs.sortedByDescending { it.score }
                    .take(15)
                    .joinToString("\n") { category ->
                        "Label: ${category.label}, Score: ${"%.2f".format(category.score * 100)}%"
                    }
            }
        }

        // ギャラリーから画像を取得
        showGalleryBtn.setOnClickListener {
            val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                addCategory(Intent.CATEGORY_OPENABLE)
                setType("image/*")
            }
            startActivityForResult(intent, GALLERY_REQUEST_CODE)
        }

        // テストデータの入力
        inputTestDataBtn.setOnClickListener {
            bitmap = getTestImgData()
            previewImg.setImageBitmap(bitmap)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        when (requestCode) {
            GALLERY_REQUEST_CODE -> {
                // ギャラリーから画像データの取得
                bitmap = MediaStore.Images.Media.getBitmap(contentResolver, data?.data ?: return)
                previewImg.setImageBitmap(bitmap)
            }
        }
    }

    /**
     * モデルをロードして画像分類を行う
     */
    private suspend fun classifyImage(targetBitmap: Bitmap) = withContext(Dispatchers.IO) {
        val model = MobilenetV2.newInstance(this@MainActivity)
        val image = TensorImage.fromBitmap(targetBitmap)
        val output = model.process(image)  // 推論
        val categoryList = output.probabilityAsCategoryList
        model.close()
        return@withContext categoryList
    }

    // テスト画像の読み込み
    private fun getTestImgData(): Bitmap {
        val randomInt = (1..11).random()
        val fileName = "test$randomInt.jpg"
        return BitmapFactory.decodeStream(assets.open(fileName))
    }
}
