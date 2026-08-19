import React from "react"
import { cn } from "../../lib/utils"

export function Tabs({ defaultValue, value, onValueChange, children, className }) {
  const [selected, setSelected] = React.useState(defaultValue || "")
  const current = value !== undefined ? value : selected
  const handleChange = onValueChange || setSelected

  return (
    <div className={cn("w-full space-y-4", className)}>
      {React.Children.map(children, (child) => {
        if (!React.isValidElement(child)) return null
        return React.cloneElement(child, {
          activeValue: current,
          onSelect: handleChange,
        })
      })}
    </div>
  )
}

export function TabsList({ activeValue, onSelect, className, children }) {
  return (
    <div
      className={cn(
        "inline-flex h-10 items-center justify-center rounded-xl bg-slate-100 dark:bg-slate-800 p-1 text-slate-500 dark:text-slate-400",
        className
      )}
    >
      {React.Children.map(children, (child) => {
        if (!React.isValidElement(child)) return null
        return React.cloneElement(child, {
          isActive: child.props.value === activeValue,
          onSelect: () => onSelect(child.props.value),
        })
      })}
    </div>
  )
}

export function TabsTrigger({ _value, isActive, onSelect, className, children, ...props }) {
  return (
    <button
      type="button"
      className={cn(
        "inline-flex items-center justify-center whitespace-nowrap rounded-lg px-3.5 py-1.5 text-sm font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 disabled:pointer-events-none disabled:opacity-50 cursor-pointer",
        isActive
          ? "bg-white dark:bg-slate-900 text-slate-900 dark:text-white shadow-sm font-semibold"
          : "hover:text-slate-900 dark:hover:text-white text-slate-600 dark:text-slate-400",
        className
      )}
      onClick={onSelect}
      {...props}
    >
      {children}
    </button>
  )
}

export function TabsContent({ value, activeValue, className, children, ...props }) {
  if (value !== activeValue) return null
  return (
    <div
      className={cn("mt-2 ring-offset-background focus-visible:outline-none", className)}
      {...props}
    >
      {children}
    </div>
  )
}
