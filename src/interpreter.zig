const std = @import("std");
const parsing = @import("./parser.zig");

const Ast = parsing.Ast;
const Parser = parsing.Parser;
const Token = parsing.Token;

fn Stack(comptime T: type) type {
    return struct {
        data: []T,
        size: usize,
        alloc: std.mem.Allocator,
        const Self = @This();

        pub fn init(initialCap: usize, alloc: std.mem.Allocator) !Self {
            return Self{
                .alloc = alloc,
                .data = try alloc.alloc(T, initialCap),
                .size = 0,
            };
        }

        pub fn deinit(self: *Self) void {
            self.alloc.free(self.data);
        }

        pub fn push(self: *Self, elem: T) !void {
            if (self.size >= self.data.len - 1) {
                self.data = try self.alloc.realloc(self.data, self.data.len * 2);
            }

            self.data[self.size] = elem;
            self.size += 1;
        }

        pub fn pop(self: *Self) ?T {
            if (self.size > 0) {
                const elem: T = self.data[self.size - 1];
                self.size -= 1;
                return elem;
            } else {
                return null;
            }
        }
    };
}

pub const Interpreter = struct {
    const ResultTag = enum { int, float, err };
    const Result = union(ResultTag) {
        int: i64,
        float: f64,
        err: struct {
            message: []const u8,
            loc: parsing.Location,
        },

        pub fn toStr(self: Result, allocator: std.mem.Allocator) []u8 {
            return switch (self) {
                .int => |val| std.fmt.allocPrint(allocator, "<int: {d}>", .{val}) catch unreachable,
                .float => |val| std.fmt.allocPrint(allocator, "<float: {d}>", .{val}) catch unreachable,
                .err => |val| std.fmt.allocPrint(allocator, "<err: {s}>", .{val.message}) catch unreachable,
            };
        }

        pub fn isZero(self: Result) bool {
            return switch (self) {
                .int => |val| val == 0,
                .float => |val| val == 0,
                .err => true,
            };
        }

        pub fn isEmpty(self: Result) bool {
            return self.isErr();
        }

        pub fn isErr(self: Result) bool {
            return @as(ResultTag, self) == ResultTag.err;
        }

        pub fn add(lhs: Result, rhs: Result) Result {
            if (lhs.isZero()) return rhs;
            if (rhs.isZero()) return lhs;

            return switch (lhs) {
                ResultTag.err => lhs,
                ResultTag.int => |lhsVal| blk: {
                    switch (rhs) {
                        ResultTag.err => break :blk rhs,
                        ResultTag.int => |rhsVal| {
                            break :blk Result{
                                .int = lhsVal + rhsVal,
                            };
                        },
                        ResultTag.float => |rhsVal| {
                            const floatLhs: f64 = @floatFromInt(lhsVal);
                            break :blk Result{
                                .float = floatLhs + rhsVal,
                            };
                        },
                    }
                },
                ResultTag.float => |lhsVal| blk: {
                    const rhsVal: f64 = switch (rhs) {
                        ResultTag.err => break :blk rhs,
                        ResultTag.int => |rhsVal| @floatFromInt(rhsVal),
                        ResultTag.float => |rhsVal| rhsVal,
                    };

                    break :blk Result{
                        .float = lhsVal + rhsVal,
                    };
                },
            };
        }
        pub fn sub(lhs: Result, rhs: Result) Result {
            if (lhs.isZero()) return rhs;
            if (rhs.isZero()) return lhs;

            return switch (lhs) {
                ResultTag.err => lhs,
                ResultTag.int => |lhsVal| blk: {
                    switch (rhs) {
                        ResultTag.err => break :blk rhs,
                        ResultTag.int => |rhsVal| {
                            break :blk Result{
                                .int = lhsVal - rhsVal,
                            };
                        },
                        ResultTag.float => |rhsVal| {
                            const floatLhs: f64 = @floatFromInt(lhsVal);
                            break :blk Result{
                                .float = floatLhs - rhsVal,
                            };
                        },
                    }
                },
                ResultTag.float => |lhsVal| blk: {
                    const rhsVal: f64 = switch (rhs) {
                        ResultTag.err => break :blk rhs,
                        ResultTag.int => |rhsVal| @floatFromInt(rhsVal),
                        ResultTag.float => |rhsVal| rhsVal,
                    };

                    break :blk Result{
                        .float = lhsVal - rhsVal,
                    };
                },
            };
        }
        pub fn mul(lhs: Result, rhs: Result) Result {
            if (lhs.isZero()) return lhs;
            if (rhs.isZero()) return rhs;
            return switch (lhs) {
                ResultTag.err => lhs,
                ResultTag.int => |lhsVal| blk: {
                    switch (rhs) {
                        ResultTag.err => break :blk rhs,
                        ResultTag.int => |rhsVal| {
                            break :blk Result{
                                .int = lhsVal * rhsVal,
                            };
                        },
                        ResultTag.float => |rhsVal| {
                            const floatLhs: f64 = @floatFromInt(lhsVal);
                            break :blk Result{
                                .float = floatLhs * rhsVal,
                            };
                        },
                    }
                },
                ResultTag.float => |lhsVal| blk: {
                    const rhsVal: f64 = switch (rhs) {
                        ResultTag.err => break :blk rhs,
                        ResultTag.int => |rhsVal| @floatFromInt(rhsVal),
                        ResultTag.float => |rhsVal| rhsVal,
                    };

                    break :blk Result{
                        .float = lhsVal * rhsVal,
                    };
                },
            };
        }
        pub fn div(lhs: Result, rhs: Result) Result {
            if (lhs.isZero() or rhs.isZero()) {
                return Result{
                    .err = .{
                        .message = "Cannot divide by zero",
                        .loc = undefined,
                    },
                };
            }

            return switch (lhs) {
                ResultTag.err => |val| Result{ .err = val },
                ResultTag.int => |lhsVal| blk: {
                    const rhsVal: f64 = switch (rhs) {
                        ResultTag.int => |rhsVal| @floatFromInt(rhsVal),
                        ResultTag.float => |rhsVal| rhsVal,
                        ResultTag.err => |val| break :blk Result{ .err = val },
                    };
                    const floatLhs: f64 = @floatFromInt(lhsVal);

                    break :blk Result{
                        .float = floatLhs / rhsVal,
                    };
                },
                ResultTag.float => |lhsVal| blk: {
                    const rhsVal: f64 = switch (rhs) {
                        ResultTag.int => |rhsVal| @floatFromInt(rhsVal),
                        ResultTag.float => |rhsVal| rhsVal,
                        ResultTag.err => |val| break :blk Result{ .err = val },
                    };

                    break :blk Result{
                        .float = lhsVal / rhsVal,
                    };
                },
            };
        }
        pub fn lsh(lhs: Result, rhs: Result) Result {
            if (lhs.isZero()) return rhs;
            if (rhs.isZero()) return lhs;

            return switch (rhs) {
                ResultTag.err => |val| Result{ .err = val },
                ResultTag.float => |val| {
                    if (val < 0 or val > 64) {
                        const message = std.fmt.allocPrint(std.heap.page_allocator, "Invalid value '{d}' for rhs of left shift.", .{val}) catch "";
                        return Result{ .err = .{
                            .message = message,
                            .loc = undefined,
                        } };
                    }
                    const rhsVal: u6 = @intFromFloat(val);

                    const lhsVal: i64 = switch (lhs) {
                        ResultTag.float => |fval| @intFromFloat(@floor(fval)),
                        ResultTag.int => |ival| ival,
                        ResultTag.err => |eval| return Result{ .err = eval },
                    };

                    return Result{
                        .int = lhsVal << rhsVal,
                    };
                },
                ResultTag.int => |val| {
                    if (val < 0 or val > 64) {
                        const message = std.fmt.allocPrint(std.heap.page_allocator, "Invalid value '{d}' for rhs of left shift.", .{val}) catch "";
                        return Result{ .err = .{
                            .message = message,
                            .loc = undefined,
                        } };
                    }

                    const rhsVal: u6 = @intCast(val);

                    const lhsVal: i64 = switch (lhs) {
                        ResultTag.float => |fval| @intFromFloat(@floor(fval)),
                        ResultTag.int => |ival| ival,
                        ResultTag.err => |eval| return Result{ .err = eval },
                    };

                    return Result{
                        .int = lhsVal << rhsVal,
                    };
                },
            };
        }
        pub fn rsh(lhs: Result, rhs: Result) Result {
            if (lhs.isZero()) return rhs;
            if (rhs.isZero()) return lhs;

            return switch (rhs) {
                ResultTag.err => |val| Result{ .err = val },
                ResultTag.float => |val| {
                    if (val < 0 or val > 64) {
                        const message = std.fmt.allocPrint(std.heap.page_allocator, "Invalid value '{d}' for rhs of left shift.", .{val}) catch "";
                        return Result{ .err = .{
                            .message = message,
                            .loc = undefined,
                        } };
                    }
                    const rhsVal: u6 = @intFromFloat(val);

                    const lhsVal: i64 = switch (lhs) {
                        ResultTag.float => |fval| @intFromFloat(@floor(fval)),
                        ResultTag.int => |ival| ival,
                        ResultTag.err => |eval| return Result{ .err = eval },
                    };

                    return Result{
                        .int = lhsVal >> rhsVal,
                    };
                },
                ResultTag.int => |val| {
                    if (val < 0 or val > 64) {
                        const message = std.fmt.allocPrint(std.heap.page_allocator, "Invalid value '{d}' for rhs of left shift.", .{val}) catch "";
                        return Result{ .err = .{
                            .message = message,
                            .loc = undefined,
                        } };
                    }

                    const rhsVal: u6 = @intCast(val);

                    const lhsVal: i64 = switch (lhs) {
                        ResultTag.float => |fval| @intFromFloat(@floor(fval)),
                        ResultTag.int => |ival| ival,
                        ResultTag.err => |eval| return Result{ .err = eval },
                    };

                    return Result{
                        .int = lhsVal >> rhsVal,
                    };
                },
            };
        }
        pub fn band(lhs: Result, rhs: Result) Result {
            if (lhs.isEmpty()) return lhs;
            if (rhs.isEmpty()) return rhs;

            const lhsVal: i64 = switch (lhs) {
                ResultTag.float => |val| @intFromFloat(val),
                ResultTag.int => |val| val,
                ResultTag.err => unreachable,
            };

            const rhsVal: i64 = switch (rhs) {
                ResultTag.float => |val| @intFromFloat(val),
                ResultTag.int => |val| val,
                ResultTag.err => unreachable,
            };

            return Result{
                .int = lhsVal & rhsVal,
            };
        }
        pub fn xor(lhs: Result, rhs: Result) Result {
            if (lhs.isEmpty()) return lhs;
            if (rhs.isEmpty()) return rhs;

            const lhsVal: i64 = switch (lhs) {
                ResultTag.float => |val| @intFromFloat(val),
                ResultTag.int => |val| val,
                ResultTag.err => unreachable,
            };

            const rhsVal: i64 = switch (rhs) {
                ResultTag.float => |val| @intFromFloat(val),
                ResultTag.int => |val| val,
                ResultTag.err => unreachable,
            };

            return Result{
                .int = lhsVal ^ rhsVal,
            };
        }
        pub fn bor(lhs: Result, rhs: Result) Result {
            if (lhs.isEmpty()) return lhs;
            if (rhs.isEmpty()) return rhs;

            const lhsVal: i64 = switch (lhs) {
                ResultTag.float => |val| @intFromFloat(val),
                ResultTag.int => |val| val,
                ResultTag.err => unreachable,
            };

            const rhsVal: i64 = switch (rhs) {
                ResultTag.float => |val| @intFromFloat(val),
                ResultTag.int => |val| val,
                ResultTag.err => unreachable,
            };

            return Result{
                .int = lhsVal | rhsVal,
            };
        }
    };

    pub const Symbol = struct {
        name: []const u8,
        val: ?Result,
    };

    stack: Stack(Symbol),
    symbolTable: std.StringHashMap(Result),
    allocator: std.mem.Allocator,

    pub fn init(alloc: std.mem.Allocator) !Interpreter {
        return .{
            .allocator = alloc,
            .stack = try Stack(Symbol).init(4, alloc),
            .symbolTable = std.StringHashMap(Result).init(alloc),
        };
    }

    pub fn deinit(self: *Interpreter) void {
        self.symbolTable.deinit();
        self.stack.deinit();
    }

    pub fn interpretAst(self: *Interpreter, ast: Ast) void {
        for (ast.head.expressions.items) |node| {
            self.interpretAstNode(node);
            while (self.stack.size > 0) {
                const res = self.stack.pop().?.val;
                if (res == null) continue;

                switch (res.?) {
                    .float => |val| std.debug.print("res: {d}\n", .{val}),
                    .int => |val| std.debug.print("res: {d}\n", .{val}),
                    .err => |val| std.debug.print("Error: '{s}'\nAt: {d}, {d}\n", .{ val.message, val.loc.row, val.loc.col }),
                }
            }
        }
        var symbolKeys = self.symbolTable.keyIterator();

        std.debug.print("Symbol table:\n", .{});
        while (symbolKeys.next()) |key| {
            const val = self.symbolTable.get(key.*).?;
            std.debug.print("    name: {s}, val: {s}\n", .{ key.*, val.toStr(self.allocator) });
        }
    }

    fn interpretAstNode(self: *Interpreter, node: Ast.Node) void {
        switch (node) {
            .binOp => |val| self.interpretBinOp(val),
            .number => |val| self.interpretNumber(val),
            .ident => |val| self.interpretIdent(val),
            .varDecl => |val| self.interpretVarDecl(val),
            .varAssign => |val| self.interpretVarAssign(val),
            .funcDecl, .ret => unreachable,
            .program => unreachable,
        }
    }

    fn interpretIdent(self: *Interpreter, node: *Ast.Node.Ident) void {
        const resMaybe = self.symbolTable.get(node.value.value);
        if (resMaybe == null) {
            std.debug.print("{d}:{d}: Ident {s} is undefined\n", .{ node.loc.row, node.loc.col, node.value.value });
        } else {
            const res = resMaybe.?;
            std.debug.print("{d}:{d}: {s} value: {s}\n", .{ node.loc.row, node.loc.col, node.value.value, res.toStr(self.allocator) });
        }
        self.stack.push(Symbol{ .name = node.value.value, .val = resMaybe }) catch |err| {
            std.debug.print("INTERNAL ERROR: Could not push indent to stack: {?}\n", .{err});
        };
    }

    fn interpretVarAssign(self: *Interpreter, node: *Ast.Node.VarAssign) void {
        self.interpretIdent(&node.ident);

        const identMaybe = self.stack.pop();

        if (identMaybe == null) {
            std.debug.print("{d}:{d}: ERROR: Internal parsing error. Could not find symbol for vardecl.\n", .{ node.loc.row, node.loc.col });
            return;
        }
        const ident = identMaybe.?;

        if (ident.val == null) {
            std.debug.print("{d}:{d}: ERROR: Assignment of undefined symbol '{s}'.\n", .{ node.loc.row, node.loc.col, ident.name });
            return;
        }

        self.interpretAstNode(node.value);

        const valSym = self.stack.pop();
        if (valSym == null or valSym.?.val == null) {
            std.debug.print("{d}:{d}: ERROR: Cannot assign void value to symbol '{s}'.\n", .{ node.loc.row, node.loc.col, ident.name });
            return;
        }

        const val = valSym.?.val.?;

        self.symbolTable.put(ident.name, val) catch |err| {
            std.debug.print("INTERNAL ERROR: Could not update symbol '{s}' in symboltable: {?}\n", .{ ident.name, err });
        };
    }

    fn interpretVarDecl(self: *Interpreter, node: *Ast.Node.VarDecl) void {
        self.interpretIdent(&node.ident);

        const identSym = self.stack.pop();

        if (identSym == null) {
            std.debug.print("{d}:{d}: ERROR: Internal parsing error. Could not find symbol for vardecl.\n", .{ node.loc.row, node.loc.col });
            return;
        }
        const ident = identSym.?;

        if (ident.val != null) {
            std.debug.print("{d}:{d}: ERROR: Redefinition of symbol '{s}'.\n", .{ node.loc.row, node.loc.col, ident.name });
        }

        self.interpretAstNode(node.value);

        const valSym = self.stack.pop();
        if (valSym == null or valSym.?.val == null) {
            std.debug.print("{d}:{d}: ERROR: Cannot assign void value to symbol '{s}'.\n", .{ node.loc.row, node.loc.col, ident.name });
            return;
        }

        const val = valSym.?.val.?;

        self.symbolTable.put(ident.name, val) catch |err| {
            std.debug.print("INTERNAL ERROR: Could not update symbol '{s}' in symboltable: {?}\n", .{ ident.name, err });
        };
    }

    fn interpretBinOp(self: *Interpreter, node: *Ast.Node.BinOp) void {
        self.interpretAstNode(node.lhs);
        const lhsSym = self.stack.pop();
        if (lhsSym == null) {
            std.debug.print("{d}:{d}: ERROR: Could not interpret left hand side of binary operation\n", .{ node.loc.row, node.loc.col });
            return;
        }
        const lhs = lhsSym.?.val;

        if (lhs.?.isErr()) {
            std.debug.print("{d}:{d}: ERROR: Could not interpret left hand side of binary operation: '{s}'\n", .{ lhs.?.err.loc.row, lhs.?.err.loc.col, lhs.?.err.message });
            return;
        }
        self.interpretAstNode(node.rhs);
        const rhsSym = self.stack.pop();
        if (rhsSym == null) {
            std.debug.print("{d}:{d}: ERROR: Could not interpret right hand side of binary operation\n", .{ node.loc.row, node.loc.col });
            return;
        }

        const rhs = rhsSym.?.val;
        if (rhs.?.isErr()) {
            std.debug.print("{d}:{d}: ERROR: Could not interpret right hand side of binary operation: '{s}'\n", .{ rhs.?.err.loc.row, rhs.?.err.loc.col, rhs.?.err.message });
            return;
        }

        self.interpretOperation(lhs.?, rhs.?, node.op);
    }

    fn interpretOperation(self: *Interpreter, lhs: Result, rhs: Result, op: Token) void {
        // std.debug.print("lhs: {any}, op: {s}, rhs: {any}\n", .{ lhs, op.value, rhs });

        var res: Result = undefined;
        if (std.mem.eql(u8, op.value, "+")) {
            res = lhs.add(rhs);
        } else if (std.mem.eql(u8, op.value, "-")) {
            res = lhs.sub(rhs);
        } else if (std.mem.eql(u8, op.value, "*")) {
            res = lhs.mul(rhs);
        } else if (std.mem.eql(u8, op.value, "/")) {
            res = lhs.div(rhs);
        } else if (op.type == .BIT_LSH) {
            res = lhs.lsh(rhs);
        } else if (op.type == .BIT_RSH) {
            res = lhs.rsh(rhs);
        } else if (op.type == .BIT_AND) {
            res = lhs.band(rhs);
        } else if (op.type == .BIT_XOR) {
            res = lhs.xor(rhs);
        } else if (op.type == .BIT_OR) {
            res = lhs.bor(rhs);
        } else {
            std.debug.panic("Unreachable in 'interpretBinOp'\n", .{});
        }

        const symbol: Symbol = switch (res) {
            .float => |val| blk: {
                break :blk Symbol{
                    .name = "float",
                    .val = Result{ .float = val },
                };
            },
            .int => |val| blk: {
                break :blk Symbol{
                    .name = "int",
                    .val = Result{ .int = val },
                };
            },
            .err => |*val| blk: {
                val.*.loc = op.loc;
                break :blk Symbol{
                    .name = "err",
                    .val = Result{ .err = val.* },
                };
            },
        };
        self.stack.push(symbol) catch |err| {
            std.debug.print("INTERNAL ERROR: Could not push result of bin op to stack: {?}\n", .{err});
        };
    }

    fn interpretNumber(self: *Interpreter, node: *Ast.Node.Number) void {
        if (node.type == .float) {
            const val = std.fmt.parseFloat(f64, node.value.value) catch {
                self.stack.push(Symbol{ .name = "error", .val = Result{ .err = .{
                    .message = "Invalid float.",
                    .loc = node.loc,
                } } }) catch |err| {
                    std.debug.print("INTERNAL ERROR: Could not push result of bin op to stack: {?}\n", .{err});
                };
                return;
            };
            self.stack.push(Symbol{ .name = "float", .val = Result{
                .float = val,
            } }) catch |err| {
                std.debug.print("INTERNAL ERROR: Could not push result of bin op to stack: {?}\n", .{err});
            };
        } else {
            const val = std.fmt.parseInt(i64, node.value.value, 10) catch {
                self.stack.push(Symbol{ .name = "error", .val = Result{ .err = .{
                    .message = "Invalid int.",
                    .loc = node.loc,
                } } }) catch |err| {
                    std.debug.print("INTERNAL ERROR: Could not push result of bin op to stack: {?}\n", .{err});
                };
                return;
            };
            self.stack.push(Symbol{ .name = "int", .val = Result{
                .int = val,
            } }) catch |err| {
                std.debug.print("INTERNAL ERROR: Could not push result of bin op to stack: {?}\n", .{err});
            };
        }
    }
};
