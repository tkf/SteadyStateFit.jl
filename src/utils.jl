function shortsummary(x; context=devnull)
    context = IOContext(
        context,
        :compact => true,
        :limit => true,
    )
    return first(sort(
        [
            sprint(summary, x; context=context),
            sprint(print, x; context=context),
        ],
        by = length,
    ))
end
