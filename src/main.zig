const std = @import("std");
const parsing = @import("./parser.zig");
const interpreter = @import("./interpreter.zig");
const symbolTableVisitor = @import("symbolTableVisitor.zig");
const pretty = @import("pretty");

const Parser = parsing.Parser;
const Interpreter = interpreter.Interpreter;
const SymbolTableVisitor = symbolTableVisitor.SymbolTableVisitor;

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    var args = try std.process.argsWithAllocator(allocator);

    _ = args.skip();

    const filepath = args.next() orelse "";

    var symTableVisitor = SymbolTableVisitor.init(allocator);
    defer symTableVisitor.deinit();

    var i: Interpreter = undefined;
    defer i.deinit();

    if (std.mem.eql(u8, filepath, "")) {
        var buffer: [1024]u8 = undefined;
        while (true) {
            @memset(buffer[0 .. buffer.len - 1], 0);
            try std.fmt.format(std.io.getStdOut(), "> ", .{});
            _ = try std.io.getStdIn().read(&buffer);
            var parser = try Parser.new(&buffer, 2, allocator);
            var ast = parser.parseInput();

            var symTable = symTableVisitor.buildSymbolTable(&ast);

            i = try Interpreter.init(allocator, &symTable);
            i.interpretAst(ast);
        }
    } else {
        const script = try std.fs.cwd().openFile(filepath, .{ .mode = .read_only });
        defer script.close();

        var buffered = std.io.bufferedReader(script.reader());
        var reader = buffered.reader();

        var buffer = std.ArrayList(u8).init(allocator);

        while (true) {
            reader.streamUntilDelimiter(buffer.writer(), '\n', null) catch |err| switch (err) {
                error.EndOfStream => break,
                else => return err,
            };
            try buffer.append('\n');
        }

        const bufferSlice = try buffer.toOwnedSlice();
        var parser = try Parser.new(bufferSlice, 2, allocator);

        var ast = parser.parseInput();

        // try pretty.print(std.heap.page_allocator, ast, .{});

        var symTable = symTableVisitor.buildSymbolTable(&ast);

        i = try Interpreter.init(allocator, &symTable);
        i.interpretAst(ast);
    }
}
