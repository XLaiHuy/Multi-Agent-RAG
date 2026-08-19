import React, { useState } from 'react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from './ui/dialog'
import { Button } from './ui/button'
import { Progress } from './ui/progress'
import { Badge } from './ui/badge'
import { Alert, AlertDescription, AlertTitle } from './ui/alert'
import { useUploadDocument, useIngestionJob } from '../hooks/useDocuments'

const STAGE_DESCRIPTIONS = {
  PARSING: 'Đang giải mã và bóc tách bố cục văn bản...',
  CHUNKING: 'Phân tách cấu trúc phân cấp Parent-Child...',
  EMBEDDING: 'Tạo vector nhúng ngữ nghĩa (Dense Embeddings)...',
  INDEXING: 'Lưu trữ chỉ mục lai (ChromaDB + BM25 Sparse)...',
  READY: 'Xử lý hoàn tất! Đã sẵn sàng tra cứu.',
  FAILED: 'Xảy ra lỗi trong quá trình xử lý tài liệu.',
}

export default function UploadModal({ open = true, onClose }) {
  const [file, setFile] = useState(null)
  const [activeJobId, setActiveJobId] = useState(null)
  const [allowedRoles, _setAllowedRoles] = useState('admin,legal,finance,hr,user')

  const uploadMutation = useUploadDocument()
  const { data: jobData, error: jobError } = useIngestionJob(activeJobId)

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      setActiveJobId(null)
      uploadMutation.reset()
    }
  }

  const handleUpload = () => {
    if (!file) return

    uploadMutation.mutate(
      { file, allowedRoles },
      {
        onSuccess: (data) => {
          if (data && data.job_id) {
            setActiveJobId(data.job_id)
          }
        },
      }
    )
  }

  const isUploading = uploadMutation.isPending
  const jobStatus = jobData?.status
  const progressPct = jobData?.progress_pct || (isUploading ? 10 : 0)
  const isDone = jobStatus === 'READY'
  const isFailed = jobStatus === 'FAILED' || uploadMutation.isError

  const errorMessage =
    uploadMutation.error?.message ||
    jobData?.error_message ||
    (jobError ? jobError.message : null)

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <div className="flex items-center space-x-2">
            <span className="text-2xl">📥</span>
            <div>
              <DialogTitle>Tải lên Hợp đồng & Hồ sơ Pháp lý</DialogTitle>
              <DialogDescription className="mt-1">
                Tự động bóc tách cấu trúc, tạo cây Parent-Child và lập chỉ mục lai (Hybrid RAG).
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>

        <div className="space-y-4 my-2">
          {/* Drag & Drop File Picker Zone */}
          <div
            className={`border-2 border-dashed rounded-xl p-6 text-center transition-all duration-200 ${
              file
                ? 'border-blue-500 bg-blue-50/50 dark:bg-blue-950/20'
                : 'border-slate-300 dark:border-slate-700 bg-slate-50/60 dark:bg-slate-900/50 hover:bg-slate-100 dark:hover:bg-slate-800/50'
            }`}
          >
            <input
              type="file"
              id="contract-file-upload"
              accept=".pdf,.txt,.md,.json,.docx,.doc,.png,.jpg,.jpeg,.webp"
              onChange={handleFileChange}
              disabled={isUploading || (Boolean(activeJobId) && !isDone && !isFailed)}
              className="hidden"
            />
            <label htmlFor="contract-file-upload" className="cursor-pointer block">
              <div className="text-4xl mb-3">📄</div>
              {file ? (
                <div>
                  <p className="font-semibold text-blue-600 dark:text-blue-400 text-sm truncate max-w-xs mx-auto">
                    {file.name}
                  </p>
                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                    {(file.size / 1024 / 1024).toFixed(2)} MB • Sẵn sàng nạp vào hệ thống
                  </p>
                </div>
              ) : (
                <div>
                  <p className="font-medium text-slate-800 dark:text-slate-200 text-sm">
                    Nhấp vào đây để chọn tệp hợp đồng
                  </p>
                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                    Hỗ trợ: PDF (Native / Scanned), Word (.docx), Markdown (.md), TXT, JSON, Image OCR
                  </p>
                </div>
              )}
            </label>
          </div>

          {/* Uploading & Ingestion Progress */}
          {(isUploading || Boolean(activeJobId)) && (
            <div className="space-y-2 p-4 rounded-xl bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
              <div className="flex items-center justify-between text-xs font-semibold">
                <span className="text-slate-700 dark:text-slate-300 flex items-center gap-1.5">
                  {!isDone && !isFailed && <span className="animate-spin">⏳</span>}
                  {isDone && '✅'}
                  {isFailed && '❌'}
                  {isUploading
                    ? 'Đang gửi tệp lên máy chủ...'
                    : STAGE_DESCRIPTIONS[jobStatus] || 'Đang xử lý...'}
                </span>
                <Badge variant={isDone ? 'success' : isFailed ? 'destructive' : 'default'}>
                  {progressPct}%
                </Badge>
              </div>
              <Progress value={progressPct} className="h-2" />
            </div>
          )}

          {/* Success Notification */}
          {isDone && (
            <Alert variant="success">
              <AlertTitle className="font-semibold text-emerald-800 dark:text-emerald-300">
                Xử lý thành công!
              </AlertTitle>
              <AlertDescription className="text-xs text-emerald-700 dark:text-emerald-400">
                Tài liệu đã được phân tách và lưu trữ vào kho Vector & BM25. Bạn có thể đặt câu hỏi ngay bây giờ.
              </AlertDescription>
            </Alert>
          )}

          {/* Error Notification */}
          {isFailed && (
            <Alert variant="destructive">
              <AlertTitle className="font-semibold">Quá trình xử lý thất bại</AlertTitle>
              <AlertDescription className="text-xs">
                {errorMessage || 'Không thể hoàn tất bóc tách dữ liệu.'}
              </AlertDescription>
            </Alert>
          )}
        </div>

        <DialogFooter className="mt-4">
          <Button variant="outline" onClick={onClose}>
            {isDone ? 'Đóng' : 'Hủy bỏ'}
          </Button>
          {!isDone && (
            <Button
              onClick={handleUpload}
              disabled={!file || isUploading || (Boolean(activeJobId) && !isFailed)}
            >
              {isUploading || (Boolean(activeJobId) && !isFailed) ? (
                <span className="flex items-center gap-2">
                  <span className="animate-spin">⏳</span> Đang nạp dữ liệu...
                </span>
              ) : (
                'Bắt đầu Tải lên & Index'
              )}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
