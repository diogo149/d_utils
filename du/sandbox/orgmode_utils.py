def df_to_org(df, **kwargs):
    lines = df.to_csv(index=False, sep="|", **kwargs).strip().split("\n")
    lines = ["|" + line for line in lines]
    lines.insert(1, "|-")
    return "\n".join(lines)
