import React from "react"
import { cn } from "../../lib/utils"

export function Progress({ value = 0, className, ...props }) {
  const percentage = Math.min(100, Math.max(0, value))
  return (
    <div
      className={cn(
        "relative h-2.5 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800",
        className
      )}
      {...props}
    >
      <div
        className="h-full bg-gradient-to-r from-blue-600 to-indigo-600 transition-all duration-300 ease-out"
        style={{ width: `${percentage}%` }}
      />
    </div>
  )
}
