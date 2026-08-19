import React, { useEffect } from "react"
import { cn } from "../../lib/utils"

export function Dialog({ open, onOpenChange, children }) {
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === "Escape" && open && onOpenChange) {
        onOpenChange(false)
      }
    }
    if (open) {
      document.body.style.overflow = "hidden"
      window.addEventListener("keydown", handleKeyDown)
    } else {
      document.body.style.overflow = "unset"
    }
    return () => {
      document.body.style.overflow = "unset"
      window.removeEventListener("keydown", handleKeyDown)
    }
  }, [open, onOpenChange])

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 overflow-y-auto">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/60 backdrop-blur-sm transition-opacity animate-in fade-in"
        onClick={() => onOpenChange && onOpenChange(false)}
      />
      {/* Content Container */}
      <div className="relative z-50 w-full max-w-lg animate-in zoom-in-95 duration-200">
        {children}
      </div>
    </div>
  )
}

export function DialogContent({ className, children, ...props }) {
  return (
    <div
      className={cn(
        "rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-6 shadow-2xl transition-all",
        className
      )}
      {...props}
    >
      {children}
    </div>
  )
}

export function DialogHeader({ className, ...props }) {
  return (
    <div
      className={cn("flex flex-col space-y-1.5 text-center sm:text-left mb-4", className)}
      {...props}
    />
  )
}

export function DialogTitle({ className, ...props }) {
  return (
    <h2
      className={cn("text-xl font-bold tracking-tight text-slate-900 dark:text-white", className)}
      {...props}
    />
  )
}

export function DialogDescription({ className, ...props }) {
  return (
    <p
      className={cn("text-sm text-slate-500 dark:text-slate-400 leading-relaxed", className)}
      {...props}
    />
  )
}

export function DialogFooter({ className, ...props }) {
  return (
    <div
      className={cn("flex flex-col-reverse sm:flex-row sm:justify-end sm:space-x-2 mt-6 gap-2 sm:gap-0", className)}
      {...props}
    />
  )
}
