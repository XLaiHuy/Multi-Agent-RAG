import React from "react"
import { cva } from "class-variance-authority"
import { cn } from "../../lib/utils"

const alertVariants = cva(
  "relative w-full rounded-xl border p-4 text-sm [&>svg~*]:pl-7 [&>svg+div]:translate-y-[-3px] [&>svg]:absolute [&>svg]:left-4 [&>svg]:top-4 [&>svg]:text-foreground",
  {
    variants: {
      variant: {
        default: "bg-white text-slate-900 dark:bg-slate-900 dark:text-slate-100 border-slate-200 dark:border-slate-800",
        destructive: "border-red-200 text-red-900 dark:border-red-900/50 dark:text-red-300 bg-red-50/50 dark:bg-red-950/20",
        warning: "border-amber-200 text-amber-900 dark:border-amber-900/50 dark:text-amber-300 bg-amber-50/50 dark:bg-amber-950/20",
        info: "border-blue-200 text-blue-900 dark:border-blue-900/50 dark:text-blue-300 bg-blue-50/50 dark:bg-blue-950/20",
        success: "border-emerald-200 text-emerald-900 dark:border-emerald-900/50 dark:text-emerald-300 bg-emerald-50/50 dark:bg-emerald-950/20",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

export function Alert({ className, variant, ...props }) {
  return (
    <div
      role="alert"
      className={cn(alertVariants({ variant }), className)}
      {...props}
    />
  )
}

export function AlertTitle({ className, ...props }) {
  return (
    <h5
      className={cn("mb-1 font-semibold leading-none tracking-tight", className)}
      {...props}
    />
  )
}

export function AlertDescription({ className, ...props }) {
  return (
    <div
      className={cn("text-sm [&_p]:leading-relaxed opacity-90", className)}
      {...props}
    />
  )
}
