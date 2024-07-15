const std = @import("std");

pub const Location = struct {
    row: i32,
    col: i32,
};

const TokenType = enum {
    INTEGER,
    FLOAT,
    IDENT,
    FN,
    RETURN,
    OPERATION,
    LPAREN,
    RPAREN,
    LCURLY,
    RCURLY,
    EQUAL,
    COLON,
    COMMA,
    SEMI,
    BIT_LSH,
    BIT_RSH,
    BIT_AND,
    BIT_XOR,
    BIT_OR,
    NO_OP,
    EOF,
};

pub const Token = struct {
    type: TokenType,
    value: []const u8,
    loc: Location,

    pub fn EOF() Token {
        return .{
            .type = .EOF,
            .value = "",
            .loc = .{ .row = 0, .col = 0 },
        };
    }
};

const Lexer = struct {
    content: []u8,
    index: usize,
    char: u8,
    row: i32,
    col: i32,

    pub fn new(content: []u8) Lexer {
        return Lexer{
            .content = content,
            .index = 0,
            .char = content[0],
            .row = 1,
            .col = 1,
        };
    }

    pub fn getNextToken(self: *Lexer) Token {
        self.skipWhitespace();

        if (self.index == self.content.len - 1) {
            return Token{
                .type = .EOF,
                .value = "",
                .loc = .{ .row = self.row, .col = self.col },
            };
        }

        if (std.ascii.isDigit(self.char)) {
            return self.readNumeric();
        }

        if (std.ascii.isAlphabetic(self.char) or self.char == '_') {
            return self.readIdent();
        }

        if (self.char == '<' and self.peek() == '<') {
            self.advance();
            return self.advanceWithToken(.BIT_LSH, "<<");
        }

        if (self.char == '>' and self.peek() == '>') {
            self.advance();
            return self.advanceWithToken(.BIT_RSH, ">>");
        }

        return switch (self.char) {
            '+', '-', '*', '/' => self.advanceWithToken(TokenType.OPERATION, toSlice(self.char, std.heap.page_allocator)),
            '(' => self.advanceWithToken(.LPAREN, toSlice(self.char, std.heap.page_allocator)),
            ')' => self.advanceWithToken(.RPAREN, toSlice(self.char, std.heap.page_allocator)),
            '{' => self.advanceWithToken(.LCURLY, toSlice(self.char, std.heap.page_allocator)),
            '}' => self.advanceWithToken(.RCURLY, toSlice(self.char, std.heap.page_allocator)),
            '&' => self.advanceWithToken(.BIT_AND, toSlice(self.char, std.heap.page_allocator)),
            '^' => self.advanceWithToken(.BIT_XOR, toSlice(self.char, std.heap.page_allocator)),
            '|' => self.advanceWithToken(.BIT_OR, toSlice(self.char, std.heap.page_allocator)),
            ';' => self.advanceWithToken(.SEMI, toSlice(self.char, std.heap.page_allocator)),
            ':' => self.advanceWithToken(.COLON, toSlice(self.char, std.heap.page_allocator)),
            ',' => self.advanceWithToken(.COMMA, toSlice(self.char, std.heap.page_allocator)),
            '=' => self.advanceWithToken(.EQUAL, toSlice(self.char, std.heap.page_allocator)),
            0 => self.advanceWithToken(.EOF, ""),
            else => self.advanceWithToken(TokenType.NO_OP, toSlice(self.char, std.heap.page_allocator)),
        };
    }

    fn readIdent(self: *Lexer) Token {
        var chars = std.ArrayList(u8).init(std.heap.page_allocator);

        while (std.ascii.isAlphabetic(self.char) or std.ascii.isDigit(self.char) or self.char == '_') {
            chars.append(self.char) catch {
                return Token.EOF();
            };
            self.advance();
        }

        const value = chars.toOwnedSlice() catch {
            return Token.EOF();
        };

        if (isKeyword(value)) {
            return self.getBuiltin(value);
        }

        return Token{ .type = .IDENT, .value = value, .loc = .{ .row = self.row, .col = self.col } };
    }

    fn getBuiltin(self: *Lexer, value: []u8) Token {
        var tokenType: TokenType = undefined;
        if (std.mem.eql(u8, value, "fn")) {
            tokenType = .FN;
        } else if (std.mem.eql(u8, value, "return")) {
            tokenType = .RETURN;
        } else {
            tokenType = .EOF;
        }

        return Token{ .type = tokenType, .value = value, .loc = .{ .row = self.row, .col = self.col } };
    }

    fn isKeyword(value: []u8) bool {
        return std.mem.eql(u8, value, "fn") or std.mem.eql(u8, value, "return");
    }

    fn readNumeric(self: *Lexer) Token {
        var chars = std.ArrayList(u8).init(std.heap.page_allocator);

        while (std.ascii.isDigit(self.char) or self.char == '.') {
            chars.append(self.char) catch {
                return Token.EOF();
            };
            self.advance();
        }

        var isFloat: bool = false;
        for (chars.items) |char| {
            if (char == '.') {
                isFloat = true;
            }
        }

        const tokenType: TokenType = if (isFloat) .FLOAT else .INTEGER;

        const value = chars.toOwnedSlice() catch {
            return Token.EOF();
        };

        return Token{ .type = tokenType, .value = value, .loc = .{ .col = self.col, .row = self.row } };
    }

    fn skipWhitespace(self: *Lexer) void {
        while (self.index < self.content.len - 1 and std.ascii.isWhitespace(self.char)) {
            self.advance();
        }
    }

    fn peek(self: *Lexer) u8 {
        if (self.index < self.content.len - 1) {
            return self.content[self.index + 1];
        }
        return 0;
    }

    fn advanceWithToken(self: *Lexer, tokenType: TokenType, value: []const u8) Token {
        const token = Token{ .type = tokenType, .value = value, .loc = .{ .col = self.col, .row = self.row } };
        self.advance();
        return token;
    }

    fn advance(self: *Lexer) void {
        if (self.index < self.content.len - 1) {
            if (self.char == '\n') {
                self.row += 1;
                self.col = 0;
            }

            self.col += 1;
            self.index += 1;
            self.char = self.content[self.index];
        } else {
            self.char = 0;
        }
    }
};

const AstNodeTag = enum {
    binOp,
    number,
    ident,
    varDecl,
    varAssign,
    funcDecl,
    ret,
    program,
};

pub const Ast = struct {
    pub const Node = union(AstNodeTag) {
        pub const BinOp = struct {
            lhs: Ast.Node,
            op: Token,
            rhs: Ast.Node,
            loc: Location,
        };
        pub const Number = struct {
            value: Token,
            type: Type,
            loc: Location,
            pub const Type = enum {
                float,
                int,
            };
        };
        pub const Ident = struct {
            value: Token,
            loc: Location,
        };
        pub const FuncArg = struct { ident: Ast.Node.Ident, type: Ast.Node.Ident };
        pub const Return = struct {
            token: Token,
            value: Ast.Node,
            loc: Location,
        };
        pub const FuncDecl = struct {
            ident: Ast.Node.Ident,
            retType: Ast.Node.Ident,
            args: std.ArrayList(Ast.Node.FuncArg),
            block: std.ArrayList(Ast.Node),
            loc: Location,
        };
        pub const VarDecl = struct {
            ident: Ast.Node.Ident,
            value: Ast.Node,
            loc: Location,
        };
        pub const VarAssign = struct {
            ident: Ast.Node.Ident,
            value: Ast.Node,
            loc: Location,
        };
        pub const Program = struct {
            expressions: std.ArrayList(Ast.Node),
        };
        binOp: *Ast.Node.BinOp,
        number: *Ast.Node.Number,
        ident: *Ast.Node.Ident,
        varDecl: *Ast.Node.VarDecl,
        varAssign: *Ast.Node.VarAssign,
        funcDecl: *Ast.Node.FuncDecl,
        ret: *Ast.Node.Return,
        program: *Ast.Node.Program,

        pub fn getTag(self: *const Ast.Node) AstNodeTag {
            return @as(AstNodeTag, self.*);
        }

        pub fn allocBinOp(lhs: *Ast.Node, op: Token, rhs: *Ast.Node, loc: Location, alloc: std.mem.Allocator) *Ast.Node.BinOp {
            var mem = alloc.create(Ast.Node.BinOp) catch {
                std.debug.panic("Could not create new BinOp ast node\n", .{});
            };
            mem.lhs = lhs.*;
            mem.op = op;
            mem.rhs = rhs.*;
            mem.loc = loc;

            return mem;
        }

        pub fn allocNumber(value: Token, numType: Ast.Node.Number.Type, loc: Location, alloc: std.mem.Allocator) *Ast.Node.Number {
            var mem = alloc.create(Ast.Node.Number) catch {
                std.debug.panic("Could not create new Number ast node\n", .{});
            };
            mem.value = value;
            mem.type = numType;
            mem.loc = loc;

            return mem;
        }

        pub fn allocIdent(value: Token, loc: Location, alloc: std.mem.Allocator) *Ast.Node.Ident {
            var mem = alloc.create(Ast.Node.Ident) catch {
                std.debug.panic("Could not create new Number ast node\n", .{});
            };
            mem.value = value;
            mem.loc = loc;

            return mem;
        }

        pub fn allocReturn(token: Token, value: Ast.Node, loc: Location, alloc: std.mem.Allocator) *Ast.Node.Return {
            var mem = alloc.create(Ast.Node.Return) catch {
                std.debug.panic("Could not create new Number ast node\n", .{});
            };
            mem.token = token;
            mem.value = value;
            mem.loc = loc;

            return mem;
        }

        pub fn allocFuncDecl(ident: *Ident, args: *std.ArrayList(Ast.Node.FuncArg), block: *std.ArrayList(Ast.Node), retType: *Ident, loc: Location, alloc: std.mem.Allocator) *Ast.Node.FuncDecl {
            var mem = alloc.create(Ast.Node.FuncDecl) catch {
                std.debug.panic("Could not create new FuncDecl ast node\n", .{});
            };
            mem.ident = ident.*;
            mem.args = args.*;
            mem.block = block.*;
            mem.retType = retType.*;
            mem.loc = loc;

            return mem;
        }

        pub fn allocVarDecl(ident: *Ident, value: *const Ast.Node, loc: Location, alloc: std.mem.Allocator) *Ast.Node.VarDecl {
            var mem = alloc.create(Ast.Node.VarDecl) catch {
                std.debug.panic("Could not create new VarDecl ast node\n", .{});
            };
            mem.ident = ident.*;
            mem.value = value.*;
            mem.loc = loc;

            return mem;
        }

        pub fn allocVarAssign(ident: *Ident, value: *const Ast.Node, loc: Location, alloc: std.mem.Allocator) *Ast.Node.VarAssign {
            var mem = alloc.create(Ast.Node.VarAssign) catch {
                std.debug.panic("Could not create new VarAssign ast node\n", .{});
            };
            mem.ident = ident.*;
            mem.value = value.*;
            mem.loc = loc;

            return mem;
        }
    };
    head: Node.Program,
};

pub const Parser = struct {
    lexer: *Lexer,
    k: usize,
    tokens: []Token,
    index: usize,
    allocator: std.mem.Allocator,

    pub fn new(input: []u8, maxTokens: comptime_int, allocator: std.mem.Allocator) !Parser {
        var heapLexer = try allocator.alloc(Lexer, 1);
        heapLexer[0] = Lexer.new(input);
        var parser = Parser{ .lexer = &heapLexer[0], .k = maxTokens, .tokens = try allocator.alloc(Token, maxTokens), .index = 0, .allocator = allocator };

        for (0..maxTokens) |i| {
            parser.tokens[i] = parser.lexer.getNextToken();
        }

        return parser;
    }

    pub fn parseInput(self: *Parser) Ast {
        const program = Ast.Node.Program{
            .expressions = self.parseBlock(.EOF),
        };
        return Ast{ .head = program };
    }

    fn parseBlock(self: *Parser, endToken: TokenType) std.ArrayList(Ast.Node) {
        var list = std.ArrayList(Ast.Node).init(self.allocator);

        while (self.current().type != endToken) {
            const expr = switch (self.current().type) {
                .IDENT => switch (self.peek().type) {
                    .COLON => self.parseDecl(),
                    .EQUAL => block: {
                        const res = self.parseVarAssign();
                        self.consume(&[_]TokenType{.SEMI});
                        break :block res;
                    },
                    else => {
                        std.debug.print("{d}:{d}: ERROR: Invalid token after identifer. Found: {any}\n", .{ self.peek().loc.row, self.peek().loc.col, self.peek().type });
                        std.process.exit(1);
                    },
                },
                .RETURN => block: {
                    const res = self.parseReturn();
                    self.consume(&[_]TokenType{.SEMI});
                    break :block res;
                },
                TokenType.LPAREN, TokenType.FLOAT, TokenType.INTEGER => block: {
                    const res = self.parseExpr();
                    self.consume(&[_]TokenType{.SEMI});
                    break :block res;
                },
                else => {
                    std.debug.print("{d}:{d}: ERROR: Invalid token. Found: {any}\n", .{ self.current().loc.row, self.current().loc.col, self.current().type });
                    std.process.exit(1);
                },
            };

            list.append(expr) catch {
                std.debug.print("Could not append new Ast node to program list. Exiting...\n", .{});
                std.process.exit(1);
            };
            //std.debug.print("Expression #{d}: {any}\n", .{ program.expressions.items.len, expr });
            //std.debug.print("Next token: {any}\n", .{self.peek()});
        }

        return list;
    }

    fn parseReturn(self: *Parser) Ast.Node {
        std.debug.assert(self.current().type == .RETURN);
        const retToken = self.current();
        self.consume(&[_]TokenType{.RETURN});

        const retVal = self.parseExpr();

        return Ast.Node{ .ret = Ast.Node.allocReturn(retToken, retVal, retToken.loc, self.allocator) };
    }

    fn parseDecl(self: *Parser) Ast.Node {
        std.debug.assert(self.current().type == .IDENT);
        const varNode = self.parseExpr();

        self.consume(&[_]TokenType{.COLON});
        self.consume(&[_]TokenType{.EQUAL});

        if (self.current().type == .FN) {
            return self.parseFuncDecl(varNode);
        } else {
            const res = self.parseVarDecl(varNode);
            self.consume(&[_]TokenType{.SEMI});
            return res;
        }
    }

    fn parseFuncDecl(self: *Parser, varNode: Ast.Node) Ast.Node {
        std.debug.assert(varNode.getTag() == .ident);
        std.debug.assert(self.current().type == .FN);

        self.consume(&[_]TokenType{.FN});

        var args = self.parseFuncArgs();

        const retType = self.parseExpr();

        self.consume(&[_]TokenType{.LCURLY});
        var block = self.parseBlock(.RCURLY);
        self.consume(&[_]TokenType{.RCURLY});

        return Ast.Node{ .funcDecl = Ast.Node.allocFuncDecl(varNode.ident, &args, &block, retType.ident, varNode.ident.loc, self.allocator) };
    }

    fn parseFuncArgs(self: *Parser) std.ArrayList(Ast.Node.FuncArg) {
        self.consume(&[_]TokenType{.LPAREN});
        var args = std.ArrayList(Ast.Node.FuncArg).init(self.allocator);

        while (self.current().type != .RPAREN) {
            args.append(self.parseFuncArg()) catch unreachable;
            if (self.current().type == .COMMA) {
                self.consume(&[_]TokenType{.COMMA});
            }
        }

        self.consume(&[_]TokenType{.RPAREN});

        return args;
    }

    fn parseFuncArg(self: *Parser) Ast.Node.FuncArg {
        _ = self.expect(&[_]TokenType{.IDENT});
        const ident = self.parseExpr();

        _ = self.expect(&[_]TokenType{.IDENT});
        const t = self.parseExpr();

        return Ast.Node.FuncArg{ .ident = ident.ident.*, .type = t.ident.* };
    }

    fn parseVarDecl(self: *Parser, varNode: Ast.Node) Ast.Node {
        // std.debug.print("parse var decl\n", .{});
        std.debug.assert(varNode.getTag() == .ident);

        const value = self.parseExpr();

        return Ast.Node{ .varDecl = Ast.Node.allocVarDecl(varNode.ident, &value, varNode.ident.loc, self.allocator) };
    }

    fn parseVarAssign(self: *Parser) Ast.Node {
        std.debug.assert(self.current().type == .IDENT);
        const varNode = self.parseExpr();

        self.consume(&[_]TokenType{.EQUAL});

        const value = self.parseExpr();

        return Ast.Node{ .varAssign = Ast.Node.allocVarAssign(varNode.ident, &value, varNode.ident.loc, self.allocator) };
    }

    fn parseExpr(self: *Parser) Ast.Node {
        // std.debug.print("parse expr\n", .{});
        var iterate = true;
        const startLoc = self.current().loc;
        var lhs = self.parseShift();

        while (iterate) {
            switch (self.current().type) {
                .BIT_AND, .BIT_XOR, .BIT_OR => {
                    const op = self.current();
                    self.consume(&[_]TokenType{ .BIT_AND, .BIT_XOR, .BIT_OR });
                    if (self.expect(&[_]TokenType{ .LPAREN, .FLOAT, .INTEGER, .IDENT })) {
                        var rhs = self.parseShift();
                        lhs = Ast.Node{
                            .binOp = Ast.Node.allocBinOp(&lhs, op, &rhs, startLoc, self.allocator),
                        };
                    }
                },
                else => iterate = false,
            }
        }

        return lhs;
    }

    fn parseShift(self: *Parser) Ast.Node {
        // std.debug.print("parse shift\n", .{});
        var iterate = true;
        const startLoc = self.current().loc;
        var lhs = self.parseAdditive();

        while (iterate) {
            switch (self.current().type) {
                .BIT_LSH, .BIT_RSH => {
                    const op = self.current();
                    self.consume(&[_]TokenType{ .BIT_LSH, .BIT_RSH });
                    if (self.expect(&[_]TokenType{ .LPAREN, .FLOAT, .INTEGER, .IDENT })) {
                        var rhs = self.parseAdditive();
                        lhs = Ast.Node{
                            .binOp = Ast.Node.allocBinOp(&lhs, op, &rhs, startLoc, self.allocator),
                        };
                    }
                },
                else => iterate = false,
            }
        }

        return lhs;
    }

    fn parseAdditive(self: *Parser) Ast.Node {
        // std.debug.print("parse additive\n", .{});
        var iterate = true;
        const startLoc = self.current().loc;
        var lhs = self.parseTerm();

        while (iterate) {
            switch (self.current().type) {
                .OPERATION => {
                    if (std.mem.eql(u8, self.current().value, "+") or std.mem.eql(u8, self.current().value, "-")) {
                        const op = self.current();
                        self.consume(&[_]TokenType{.OPERATION});
                        if (self.expect(&[_]TokenType{ .LPAREN, .FLOAT, .INTEGER, .IDENT })) {
                            var rhs = self.parseTerm();
                            lhs = Ast.Node{
                                .binOp = Ast.Node.allocBinOp(&lhs, op, &rhs, startLoc, self.allocator),
                            };
                        }
                    } else iterate = false;
                },
                else => iterate = false,
            }
        }

        return lhs;
    }

    fn parseTerm(self: *Parser) Ast.Node {
        // std.debug.print("parse term\n", .{});
        var iterate = true;
        const startLoc = self.current().loc;
        var lhs = self.parseFactor();

        while (iterate) {
            switch (self.current().type) {
                .OPERATION => {
                    if (std.mem.eql(u8, self.current().value, "*") or std.mem.eql(u8, self.current().value, "/")) {
                        const op = self.current();
                        self.consume(&[_]TokenType{.OPERATION});
                        if (self.expect(&[_]TokenType{ .LPAREN, .FLOAT, .INTEGER, .IDENT })) {
                            var rhs = self.parseFactor();
                            lhs = Ast.Node{
                                .binOp = Ast.Node.allocBinOp(&lhs, op, &rhs, startLoc, self.allocator),
                            };
                        }
                    } else iterate = false;
                },
                else => iterate = false,
            }
        }

        return lhs;
    }

    fn parseFactor(self: *Parser) Ast.Node {
        // std.debug.print("parse factor\n", .{});
        const token = self.current();
        var res: Ast.Node = undefined;

        if (token.type == .LPAREN) {
            self.consume(&[_]TokenType{.LPAREN});
            res = self.parseExpr();
        } else if (token.type == .FLOAT) {
            res = Ast.Node{ .number = Ast.Node.allocNumber(token, .float, token.loc, self.allocator) };
        } else if (token.type == .INTEGER) {
            res = Ast.Node{ .number = Ast.Node.allocNumber(token, .int, token.loc, self.allocator) };
        } else if (token.type == .IDENT) {
            res = Ast.Node{ .ident = Ast.Node.allocIdent(token, token.loc, self.allocator) };
        } else if (token.type == .NO_OP) {
            self.consume(&[_]TokenType{.NO_OP});
            return self.parseFactor();
        }
        self.consume(&[_]TokenType{ .RPAREN, .FLOAT, .INTEGER, .IDENT });
        // std.debug.print("Factor: {any}\n", .{res});
        return res;
    }

    fn current(self: *Parser) Token {
        return self.tokens[self.index % self.k];
    }

    fn peek(self: *Parser) Token {
        return self.tokens[(self.index + 1) % self.k];
    }

    fn consume(self: *Parser, expected: []const TokenType) void {
        _ = self.expect(expected);

        self.tokens[self.index % self.k] = self.lexer.getNextToken();
        self.index += 1;

        // std.debug.print("next token: {any} '{s}'\n", .{ self.current().type, self.current().value });
    }

    fn expect(self: *Parser, expected: []const TokenType) bool {
        var validToken: bool = false;
        for (0..expected.len) |i| {
            if (self.current().type == expected[i]) {
                validToken = true;
            }
        }
        if (!validToken) {
            std.debug.print("{d}:{d}: ERROR: Invalid token. Found: {any}, expected: {any}\n", .{ self.current().loc.row, self.current().loc.col, self.current().type, expected });
            std.process.exit(1);
        }
        return validToken;
    }
};

fn toSlice(char: u8, allocator: std.mem.Allocator) []const u8 {
    const slice = allocator.alloc(u8, 1) catch unreachable;
    slice[0] = char;
    return slice;
}
