import { useState, useCallback, useEffect } from 'react'
import Button from '@/components/ui/Button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter
} from '@/components/ui/Dialog'
import { toast } from 'sonner'
import { errorMessage } from '@/lib/utils'
import { retrySelectedDocuments } from '@/api/lightrag'

import { RotateCcwIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

interface RetryDocumentsDialogProps {
  selectedDocIds: string[]
  onDocumentsRetried?: () => Promise<void>
}

export default function RetryDocumentsDialog({ selectedDocIds, onDocumentsRetried }: RetryDocumentsDialogProps) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const [isRetrying, setIsRetrying] = useState(false)

  useEffect(() => {
    if (!open) {
      setIsRetrying(false)
    }
  }, [open])

  const handleRetry = useCallback(async () => {
    if (selectedDocIds.length === 0) return

    setIsRetrying(true)
    try {
      const result = await retrySelectedDocuments(selectedDocIds)

      if (result.status === 'retry_started') {
        toast.success(t('documentPanel.retryDocuments.success', { count: result.doc_count }))
      } else if (result.status === 'busy') {
        toast.error(t('documentPanel.retryDocuments.busy'))
        setIsRetrying(false)
        return
      } else if (result.status === 'no_documents') {
        toast.warning(t('documentPanel.retryDocuments.noDocuments'))
        setIsRetrying(false)
        return
      }

      if (onDocumentsRetried) {
        onDocumentsRetried().catch(console.error)
      }

      setOpen(false)
    } catch (err) {
      toast.error(t('documentPanel.retryDocuments.error', { error: errorMessage(err) }))
    } finally {
      setIsRetrying(false)
    }
  }, [selectedDocIds, setOpen, t, onDocumentsRetried])

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button
          variant="outline"
          side="bottom"
          tooltip={t('documentPanel.retryDocuments.tooltip', { count: selectedDocIds.length })}
          size="sm"
        >
          <RotateCcwIcon /> {t('documentPanel.retryDocuments.button')}
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md" onCloseAutoFocus={(e) => e.preventDefault()}>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 font-bold">
            <RotateCcwIcon className="h-5 w-5" />
            {t('documentPanel.retryDocuments.title')}
          </DialogTitle>
          <DialogDescription className="pt-2">
            {t('documentPanel.retryDocuments.description', { count: selectedDocIds.length })}
          </DialogDescription>
        </DialogHeader>

        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)} disabled={isRetrying}>
            {t('common.cancel')}
          </Button>
          <Button
            onClick={handleRetry}
            disabled={isRetrying}
          >
            {isRetrying ? t('documentPanel.retryDocuments.retrying') : t('documentPanel.retryDocuments.confirmButton')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
