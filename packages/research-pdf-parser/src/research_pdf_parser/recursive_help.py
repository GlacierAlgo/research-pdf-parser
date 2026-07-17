"""Click group with compact recursive command discovery."""

from __future__ import annotations

import click


class RecursiveHelpGroup(click.Group):
    """Display nested command groups and preserve declaration order."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        return list(self.commands)

    def format_options(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        self.format_commands(ctx, formatter)

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        commands: list[tuple[str, click.Command, str]] = []
        for name in self.list_commands(ctx):
            command = self.get_command(ctx, name)
            if command is not None:
                commands.append((name, command, command.get_short_help_str(limit=200)))

        simple = [item for item in commands if not isinstance(item[1], click.Group)]
        groups = [item for item in commands if isinstance(item[1], click.Group)]

        if simple:
            with formatter.section("Commands"):
                formatter.write_dl([(name, help_text) for name, _, help_text in simple])

        if groups:
            with formatter.section("Command Groups"):
                for group_name, group, group_help in groups:
                    formatter.write_text(f"\n● {group_name}")
                    formatter.indent()
                    formatter.write_text(group_help)
                    names = group.list_commands(ctx)
                    rows: list[tuple[str, str]] = []
                    for index, name in enumerate(names):
                        command = group.get_command(ctx, name)
                        if command is None:
                            continue
                        branch = "└─" if index == len(names) - 1 else "├─"
                        rows.append((f"{branch} {name}", command.get_short_help_str(limit=200)))
                    if rows:
                        formatter.write_dl(rows)
                    formatter.dedent()
