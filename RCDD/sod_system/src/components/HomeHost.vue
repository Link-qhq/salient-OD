<template>
  <div class="common-layout">
    <el-container>
      <el-header>
        <div class="text">铁轨表面缺陷检测系统</div>
      </el-header>
      <el-container>
        <el-aside class="aside">
          <el-card class="aside-card" shadow="always">
            <template #header>
              <div class="card-header">
                <span>模型选择</span>
              </div>
            </template>
            <el-select
              v-model="modelName"
              placeholder="Select"
              size="large"
              style="width: 240px"
            >
              <el-option
                v-for="item in modelOptions"
                :key="item.value"
                :label="item.label"
                :value="item.value"
              />
            </el-select>
          </el-card>
          <el-card class="aside-card" shadow="always">
            <template #header>
              <div class="card-header">
                <span>图像检测步骤</span>
              </div>
            </template>
            <div style="height: 150px">
              <el-steps direction="vertical" :active="step">
                <el-step title="选择模型" />
                <el-step title="上传图像检测中" />
                <el-step title="查看检测结果" />
              </el-steps>
            </div>
          </el-card>
        </el-aside>
        <el-container>
          <el-main style="margin-top: -20px;">
            <el-card>
              <div class="upload-container">
                <div class="image-container">
                  <div class="image-wrapper">
                    <el-image
                      :src="originImageUrl"
                      fit="contain"
                      style="width: 100%; height: 75px; border-radius: 10px;"
                    />
                    <div class="image-text">
                      <span>待检测图像</span>
                    </div>
                  </div>
                  <div class="image-wrapper">
                    <el-image
                      :src="resultImageUrl"
                      fit="contain"
                      style="width: 100%; height: 75px; border-radius: 10px;"
                      :preview-src-list="[resultImageUrl]"
                    >
                    <template #toolbar="{ actions, activeIndex }">
                      <el-icon @click="actions('zoomOut')"><ZoomOut /></el-icon>
                      <el-icon
                        @click="actions('zoomIn', { enableTransition: false, zoomRate: 2 })"
                      >
                        <ZoomIn />
                      </el-icon>
                      <el-icon
                        @click="
                          actions('clockwise', { rotateDeg: 180, enableTransition: false })
                        "
                      >
                        <RefreshRight />
                      </el-icon>
                      <el-icon @click="actions('anticlockwise')"><RefreshLeft /></el-icon>
                      <el-icon @click="download(activeIndex)"><Download /></el-icon>
                    </template>  
                  </el-image>
                    <div class="image-text">
                      <span>检测结果图像</span>
                    </div>
                  </div>
                </div>
                <div class="btn-container">
                  <el-upload
                    action=""
                    :auto-upload="false"
                    :on-change="handleFileChange"
                    :before-upload="onBeforeUpload"
                    :on-success="onSuccess"
                    accept="image/*"
                    :show-file-list="false"
                  >
                    <el-button type="primary"><el-icon><Upload /></el-icon>上传图像</el-button>
                  </el-upload>
                  <el-button type="primary" @click="onClick"><el-icon><Tools /></el-icon>开始检测</el-button>
                </div>
              </div>
            </el-card>
          </el-main>
          <el-footer style="height: 200px">
            <el-card style="height: 100%;  overflow: auto;">
              <template #header>日志输出</template>
              <el-form :model="result" label-width="120" label-suffix="：" style="margin-top: -20px;">
                <el-form-item label="时间" prop="time">
                  <el-text>{{ '-' || dayjs(result.time).format('YYYY-MM-DD hh:mm:ss') }}</el-text>
                </el-form-item>
                <el-form-item label="算法模型" prop="model">
                  <el-text>{{ result.model }}</el-text>
                </el-form-item>
                <el-form-item label="完成状态" prop="info">
                  <el-text>{{ result.finish }}</el-text>
                </el-form-item>
                <el-form-item label="日志信息" prop="info">
                  <el-text>{{ result.info }}</el-text>
                </el-form-item>
              </el-form>
            </el-card>
          </el-footer>
        </el-container>
      </el-container>
    </el-container>
  </div>
</template>
<script setup lang="js">
  import service from '@/request'
import dayjs from 'dayjs'
  import { ElMessageBox } from 'element-plus'
  import { ref, computed } from 'vue'
  const modelOptions = [
    { label: 'EGOENet', value: 'EGOENet'},
    { label: 'MAAENet', value: 'MAAENet'} 
  ]
  const modelName = ref(null)
  const step = computed(() => {
    if (isSuccess.value) return 3
    if (originImageUrl.value) return 2
    return 1
  })
  const originImageUrl = ref(null);
  const resultImageUrl = ref(null);
  const isSuccess = ref(false)
  const result = ref({
    // model: 'MAAENet',
    // finish: '检测完成',
    // info: new Date() + ' load model achieve and detect image finish'
  })
  const uploadFile = ref(null)

  // 处理文件选择
  const handleFileChange = (file) => {
    // if (!validateFile(file)) return
    uploadFile.value = file.raw
    originImageUrl.value = URL.createObjectURL(file.raw)
    // previewImage.value = URL.createObjectURL(file.raw)
  }
  const onBeforeUpload = (file) => {
    console.log('上传前')
    originImageUrl.value = URL.createObjectURL(file)
    // resultImageUrl.value = URL.createObjectURL(file)
    uploadFile.value = file.raw
    return false
  }
  const onSuccess = () => {
    // 这里你可以添加图片上传成功后的逻辑
    // originImageUrl.value = URL.createObjectURL(file.raw);
    console.log('上传成功')
  };
  const onClick = () => {
    if (!modelName.value) {
      ElMessageBox.alert('请先选择模型', '提示', {
        confirmButtonText: '确认',
        type: 'error',
      })
      return
    }
    if (!originImageUrl.value) {
      ElMessageBox.alert('请先上传待检测图像', '提示', {
        confirmButtonText: '确认',
        type: 'error',
      })
      return
    }
    const formData = new FormData()
    formData.append('image', uploadFile.value)
    formData.append('user_id', 'admin')
    // 这里调用后端接口上传
    service.post('/api/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }}
    ).then((res) => {
      // result.value = res
      // isSuccess.value = true
      // 上传成功 返回结果图片
      resultImageUrl.value = res.data.result_image
      // service.post('/api/result', {
      //   name: 'rail_1.jpg'
      // }, {
      //   headers: {
      //     responseType: 'arraybuffer',
      //   }
      // }).then(res => {
      //   console.log(res)
      //   resultImageUrl.value = URL.createObjectURL(new Blob([res], { type: 'image/png' }))
      // })
    })
  }
  // const api = () => {
  //   return new Promise(resolve => {
  //     setTimeout(() => {
  //       resolve({
  //         time: new Date(),
  //         info: 'Finished',
  //         MAE: 0.025231,
  //         S: 0.955455,
  //         F: 0.892332,
  //         E: 0.923231,
  //       })
  //     }, 2000);
  //   })
  // }
  const download = () => {
  // const url = srcList[index]
  // const suffix = url.slice(url.lastIndexOf('.'))
  const filename = Date.now() + '.jpg'

  fetch('xxxx')
    .then((response) => response.blob())
    .then((blob) => {
      const blobUrl = URL.createObjectURL(new Blob([blob]))
      const link = document.createElement('a')
      link.href = blobUrl
      link.download = filename
      document.body.appendChild(link)
      link.click()
      URL.revokeObjectURL(blobUrl)
      link.remove()
    })
}
</script>
<style scoped>
::v-deep .el-form-item {
  margin-bottom: 0;
}
.common-layout {
  padding: 20px;
}
.text {
  font-size: 30px;
  font-weight: 900;
  text-align: center;
  letter-spacing: 20px;
}
.image-text {
  margin-top: 5px;
  text-align: center;
  height: 20px;
  border-radius: 10px;
  padding: 5px;
  color: #303133;
  background-color: #F2F3F5;
}
.aside {
  display: flex;
  flex-direction: column;
  gap: 20px;
}
.aside-card {
  max-width: 480px;
  height: 255px;
}
.card-header {
  text-align: center;
}
.upload-container {
  display: flex;
  flex-direction: column;
}
.btn-container {
  margin: 20px;
  gap: 50px;
  display: flex;
  /* flex-direction: column; */
  justify-content: center;
}
.image-container {
  display: flex;
  flex-direction: column;
  margin-top: 20px;
  gap: 20px;
}
.image-wrapper {
  width: 100%;
  display: flex;
  flex-direction: column;
}
</style>