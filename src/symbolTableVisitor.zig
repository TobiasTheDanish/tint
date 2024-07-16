const std = @import("std");
const parser = @import("parser.zig");
const symbolTable = @import("symbolTable.zig");

const Ast = parser.Ast;
const SymbolTable = symbolTable.SymbolTable;
const Symbol = symbolTable.Symbol;

pub const SymbolTableVisitor = struct {
    table: SymbolTable,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) SymbolTableVisitor {
        return .{
            .allocator = allocator,
            .table = SymbolTable.init(null, allocator),
        };
    }

    pub fn deinit(self: *SymbolTableVisitor) void {
        self.table.deinit();
    }

    pub fn buildSymbolTable(self: *SymbolTableVisitor, ast: *Ast) SymbolTable {
        for (ast.head.expressions.items) |node| {
            _ = self.visitNode(node);
        }

        return self.table;
    }

    fn visitNode(self: *SymbolTableVisitor, node: Ast.Node) []const u8 {
        return switch (node) {
            .funcDecl => |val| self.visitFuncDecl(val),
            .varDecl => |val| self.visitVarDecl(val),
            .binOp => |val| self.visitBinOp(val),
            .ident => |val| self.visitIdent(val),
            .number => |val| self.visitNumber(val),
            .varAssign, .ret => self.noOp(),
            else => unreachable,
        };
    }

    fn noOp(_: *SymbolTableVisitor) []const u8 {
        return "void";
    }

    fn visitBinOp(self: *SymbolTableVisitor, node: *Ast.Node.BinOp) []const u8 {
        const lhs = self.visitNode(node.lhs);
        const rhs = self.visitNode(node.rhs);

        if (std.mem.eql(u8, lhs, "void") or std.mem.eql(u8, rhs, "void")) {
            std.debug.print("{d}:{d}: ERROR: Invalid types in binary operation '{s}' {s} '{s}'\n", .{ node.op.loc.row, node.op.loc.col, lhs, node.op.value, rhs });
            return "void";
        }

        if (!std.mem.eql(u8, lhs, rhs)) {
            std.debug.print("{d}:{d}: ERROR: Invalid types in binary operation '{s}' {s} '{s}'\n", .{ node.op.loc.row, node.op.loc.col, lhs, node.op.value, rhs });
            return "void";
        }

        return lhs;
    }

    fn visitVarDecl(self: *SymbolTableVisitor, node: *Ast.Node.VarDecl) []const u8 {
        const identType = self.visitIdent(&node.ident);
        if (!std.mem.eql(u8, identType, "void")) {
            std.debug.print("{d}:{d}: ERROR: Redefinition of symbol {s}\n", .{ node.loc.row, node.loc.col, node.ident.value.value });
            return "void";
        }

        const valueType = self.visitNode(node.value);

        if (std.mem.eql(u8, valueType, "void")) {
            std.debug.print("{d}:{d}: ERROR: Cannot assign void to symbol '{s}'.\n", .{ node.loc.row, node.loc.col, node.ident.value.value });
            return "void";
        }

        self.table.addSymbol(.{ .name = node.ident.value.value, .type = valueType, .val = null });

        return "void";
    }

    fn visitFuncDecl(self: *SymbolTableVisitor, node: *Ast.Node.FuncDecl) []const u8 {
        const identType = self.visitIdent(&node.ident);
        if (!std.mem.eql(u8, identType, "void")) {
            std.debug.print("{d}:{d}: ERROR: Redefinition of symbol {s}\n", .{ node.loc.row, node.loc.col, node.ident.value.value });
            return "void";
        }

        var funcScope = SymbolTable.init(&self.table, self.allocator);

        for (node.args.items) |arg| {
            funcScope.addSymbol(.{ .name = arg.ident.value.value, .type = arg.type.value.value, .val = null });
        }

        // TODO: store funcScope as val of funcSym
        const funcSym = Symbol{ .name = node.ident.value.value, .type = node.retType.value.value, .val = null };

        self.table.addSymbol(funcSym);

        self.table = funcScope;

        for (node.block.items) |expr| {
            _ = self.visitNode(expr);
        }

        self.table = self.table.parent.?.*;

        return "void";
    }

    fn visitIdent(self: *SymbolTableVisitor, node: *Ast.Node.Ident) []const u8 {
        const maybeSymbol = self.table.getSymbol(node.value.value);

        if (maybeSymbol == null) {
            return "void";
        }

        const symbol = maybeSymbol.?;
        return symbol.type;
    }

    fn visitNumber(_: *SymbolTableVisitor, node: *Ast.Node.Number) []const u8 {
        return switch (node.type) {
            .float => "float",
            .int => "int",
        };
    }
};
